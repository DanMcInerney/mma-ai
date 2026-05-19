#!/usr/bin/env python3
import asyncio
import argparse
import csv
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import quote

import psycopg2
from playwright.async_api import async_playwright, Page, TimeoutError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("espn_scraper.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Database connection parameters
DB_PARAMS = {
    "dbname": "mma-ai",  # Update this as needed
    "user": "postgres",  # Update this as needed
    "password": os.getenv("POSTGRES_PASSWORD", ""),  # Configure locally if needed
    "host": "localhost",  # Update this as needed
    "port": "5432"  # Update this as needed
}

# Scraper settings
MAX_WORKERS = 10
REQUEST_TIMEOUT = 30  # seconds
WAIT_BETWEEN_REQUESTS = 1  # seconds to wait between requests
MAX_RETRIES = 3

class ESPNScraper:
    def __init__(self, db_params: Dict[str, str], max_workers: int = 10, output_file: str = "fighter_urls.csv", checkpoint_interval: int = 50):
        self.db_params = db_params
        self.max_workers = max_workers
        self.browser = None
        self.context = None
        self.semaphore = asyncio.Semaphore(max_workers)
        self.output_file = output_file
        self.results = []  # To store results for CSV output
        self.processed_count = 0
        self.total_count = 0
        self.lock = asyncio.Lock()  # For thread-safe counter updates
        self.checkpoint_interval = checkpoint_interval
        self.processed_fighter_ids = set()  # Track already processed fighter IDs
        
    async def initialize(self):
        """Initialize the Playwright browser and context."""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        
    async def close(self):
        """Close the browser and context."""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
            
    def get_db_connection(self):
        """Create a connection to the database."""
        return psycopg2.connect(**self.db_params)
    
    def get_all_fighters(self) -> List[Tuple[int, str]]:
        """Get all fighters from the database."""
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT fighter_id, fighter_name FROM features.fighter_mapping")
                return cur.fetchall()
    
    def get_fighter_by_name(self, fighter_name: str) -> Optional[Tuple[int, str]]:
        """Get a specific fighter by name."""
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT fighter_id, fighter_name FROM features.fighter_mapping WHERE LOWER(fighter_name) = LOWER(%s)",
                    (fighter_name,)
                )
                result = cur.fetchone()
                return result if result else None
    
    async def search_fighter(self, page: Page, fighter_name: str) -> Optional[str]:
        """Search for a fighter on ESPN and get their profile URL."""
        search_url = f"https://www.espn.com/search/_/q/{quote(fighter_name)}"
        
        for attempt in range(MAX_RETRIES):
            try:
                await page.goto(search_url, timeout=REQUEST_TIMEOUT * 1000)
                
                # Wait for search results to load
                await page.wait_for_load_state("networkidle", timeout=REQUEST_TIMEOUT * 1000)
                
                # Try to find the fighter card with various selectors
                selector = "a[href*='/mma/fighter/_/id/']"
                
                fighter_link = None
                try:
                    # Check if selector exists
                    if await page.query_selector(selector):
                            if selector.startswith("a["):
                                fighter_link = await page.query_selector(selector)
                            else:
                                # Find the anchor inside the container
                                fighter_link = await page.query_selector(f"{selector} a")
                            
                            if fighter_link:
                                break
                except Exception as e:
                    logger.debug(f"Selector {selector} failed: {e}")
                    continue
                
                if fighter_link:
                    href = await fighter_link.get_attribute("href")
                    if href and "/mma/fighter/" in href:
                        if not href.startswith("http"):
                            href = f"https://www.espn.com{href}"
                        return href
                
                # If selectors fail, take a screenshot for debugging
                await page.screenshot(path=f"debug_{fighter_name.replace(' ', '_')}.png")
                
                logger.warning(f"No fighter profile found for {fighter_name}")
                return None
                
            except TimeoutError:
                logger.warning(f"Timeout on attempt {attempt+1} for {fighter_name}")
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Failed after {MAX_RETRIES} attempts for {fighter_name}")
                    return None
                # Wait before retrying
                await asyncio.sleep(WAIT_BETWEEN_REQUESTS * (attempt + 1))
            
            except Exception as e:
                logger.error(f"Error searching for {fighter_name}: {str(e)}")
                return None
    
    def get_fighter_stats_url(self, profile_url: str) -> str:
        """Convert a fighter profile URL to their stats URL."""
        return profile_url.replace("/mma/fighter/", "/mma/fighter/stats/")
    
    async def process_fighter(self, fighter_id: int, fighter_name: str):
        """Process a single fighter - search and find their stats URL."""
        async with self.semaphore:
            page = await self.context.new_page()
            try:
                logger.info(f"Processing fighter: {fighter_name} (ID: {fighter_id})")
                
                # Search for the fighter
                profile_url = await self.search_fighter(page, fighter_name)
                if not profile_url:
                    logger.warning(f"Could not find ESPN profile for {fighter_name}")
                    self.results.append({
                        "fighter_id": fighter_id,
                        "fighter_name": fighter_name,
                        "profile_url": "",
                        "stats_url": "",
                        "status": "not found"
                    })
                    
                    # Update progress
                    await self.update_progress()
                    return
                
                # Get the stats URL
                stats_url = self.get_fighter_stats_url(profile_url)
                
                logger.info(f"Found stats URL for {fighter_name}: {stats_url}")
                
                # Verify stats URL is accessible
                stats_accessible = False
                try:
                    await page.goto(stats_url, timeout=REQUEST_TIMEOUT * 1000)
                    await page.wait_for_load_state("networkidle", timeout=REQUEST_TIMEOUT * 1000)
                    logger.info(f"Successfully accessed stats page for {fighter_name}")
                    stats_accessible = True
                except Exception as e:
                    logger.error(f"Could not access stats page for {fighter_name}: {str(e)}")
                
                # Save the result
                self.results.append({
                    "fighter_id": fighter_id,
                    "fighter_name": fighter_name,
                    "profile_url": profile_url,
                    "stats_url": stats_url,
                    "status": "success" if stats_accessible else "stats page error"
                })
                
                # Update progress
                await self.update_progress()
                
            except Exception as e:
                logger.error(f"Error processing fighter {fighter_name}: {str(e)}")
                self.results.append({
                    "fighter_id": fighter_id,
                    "fighter_name": fighter_name,
                    "profile_url": "",
                    "stats_url": "",
                    "status": f"error: {str(e)}"
                })
                
                # Update progress
                await self.update_progress()
            finally:
                await page.close()
    
    async def update_progress(self):
        """Update and log the progress."""
        async with self.lock:
            self.processed_count += 1
            
            # Log progress at regular intervals
            if self.processed_count % 10 == 0 or self.processed_count == self.total_count:
                logger.info(f"Progress: {self.processed_count}/{self.total_count} fighters processed ({self.processed_count/self.total_count*100:.1f}%)")
            
            # Save checkpoint at regular intervals
            if self.processed_count % self.checkpoint_interval == 0:
                logger.info(f"Saving checkpoint at {self.processed_count} fighters...")
                self.save_results_to_csv(is_checkpoint=True)
    
    def save_results_to_csv(self, is_checkpoint: bool = False):
        """Save the collected fighter URLs to a CSV file."""
        if not self.results:
            logger.warning("No results to save")
            return
            
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(self.output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Determine the filename to use
            filename = self.output_file
            if is_checkpoint:
                # Add timestamp to checkpoint files
                base, ext = os.path.splitext(self.output_file)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"{base}_checkpoint_{timestamp}{ext}"
            
            # Write results to CSV
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ["fighter_id", "fighter_name", "profile_url", "stats_url", "status"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for result in self.results:
                    writer.writerow(result)
                    
            logger.info(f"Saved {len(self.results)} fighter URL results to {filename}")
            
            # Print summary statistics for final save (not checkpoints)
            if not is_checkpoint:
                statuses = {}
                for result in self.results:
                    status = result["status"]
                    statuses[status] = statuses.get(status, 0) + 1
                
                logger.info("Result summary:")
                for status, count in statuses.items():
                    logger.info(f"  {status}: {count} ({count/len(self.results)*100:.1f}%)")
            
        except Exception as e:
            logger.error(f"Error saving results to CSV: {str(e)}")
    
    def load_previous_results(self, resume_file: str) -> Set[int]:
        """Load results from a previous run for resuming."""
        processed_ids = set()
        try:
            if os.path.exists(resume_file):
                with open(resume_file, 'r', newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    
                    for row in reader:
                        try:
                            fighter_id = int(row["fighter_id"])
                            processed_ids.add(fighter_id)
                            
                            # Add to our results as well
                            self.results.append({
                                "fighter_id": fighter_id,
                                "fighter_name": row["fighter_name"],
                                "profile_url": row["profile_url"],
                                "stats_url": row["stats_url"],
                                "status": row["status"]
                            })
                        except (ValueError, KeyError) as e:
                            logger.warning(f"Error reading row from resume file: {e}")
                
                logger.info(f"Loaded {len(processed_ids)} fighter results from {resume_file}")
            else:
                logger.info(f"Resume file {resume_file} does not exist. Starting fresh.")
        except Exception as e:
            logger.error(f"Error loading previous results: {e}")
        
        self.processed_fighter_ids = processed_ids
        return processed_ids

    async def run(self, fighter_name: Optional[str] = None, limit: Optional[int] = None, resume_file: Optional[str] = None):
        """Run the scraper for all fighters or a specific fighter."""
        try:
            await self.initialize()
            
            # Load previous results if resuming
            processed_ids = set()
            if resume_file:
                processed_ids = self.load_previous_results(resume_file)
                logger.info(f"Resuming scrape, skipping {len(processed_ids)} previously processed fighters")
            
            # If fighter name is provided, only process that fighter
            if fighter_name:
                fighter = self.get_fighter_by_name(fighter_name)
                if fighter:
                    fighter_id, name = fighter
                    # Skip if already processed during resume
                    if fighter_id in processed_ids:
                        logger.info(f"Skipping already processed fighter: {name} (ID: {fighter_id})")
                        return
                        
                    logger.info(f"Processing single fighter: {name} (ID: {fighter_id})")
                    self.total_count = 1
                    await self.process_fighter(fighter_id, name)
                else:
                    logger.error(f"Fighter not found: {fighter_name}")
                return
            
            # Get all fighters
            all_fighters = self.get_all_fighters()
            total_fighters = len(all_fighters)
            logger.info(f"Found {total_fighters} total fighters")
            
            # Filter out already processed fighters if resuming
            if processed_ids:
                fighters = [(fid, name) for fid, name in all_fighters if fid not in processed_ids]
                logger.info(f"Filtered to {len(fighters)} fighters after removing already processed ones")
            else:
                fighters = all_fighters
            
            # Limit the number of fighters if specified
            if limit and limit > 0:
                fighters = fighters[:limit]
                logger.info(f"Limited to processing {len(fighters)} fighters")
            
            self.total_count = len(fighters) + len(processed_ids)
            self.processed_count = len(processed_ids)
            
            if not fighters:
                logger.info("No new fighters to process")
                return
                
            # Create a task for each fighter
            tasks = []
            for fighter_id, fighter_name in fighters:
                tasks.append(self.process_fighter(fighter_id, fighter_name))
            
            # Run all tasks
            await asyncio.gather(*tasks)
            
            # Save final results to CSV
            self.save_results_to_csv()
            
        finally:
            await self.close()

async def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Scrape fighter stats from ESPN")
    parser.add_argument("-f", "--fighter", type=str, help="Scrape a specific fighter by name")
    parser.add_argument("-l", "--limit", type=int, help="Limit the number of fighters to process")
    parser.add_argument("-o", "--output", type=str, default="fighter_urls.csv", help="Output CSV file path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("-c", "--checkpoint", type=int, default=50, help="Save checkpoint every N fighters")
    parser.add_argument("-r", "--resume", type=str, help="Resume from a previous CSV file")
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    start_time = time.time()
    scraper = ESPNScraper(
        DB_PARAMS, 
        max_workers=MAX_WORKERS, 
        output_file=args.output,
        checkpoint_interval=args.checkpoint
    )
    await scraper.run(fighter_name=args.fighter, limit=args.limit, resume_file=args.resume)
    elapsed = time.time() - start_time
    logger.info(f"Scraping completed in {elapsed:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main()) 
