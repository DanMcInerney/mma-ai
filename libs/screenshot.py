import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import glob
from PIL import Image
import re
import numpy as np
from libs.paths import picks_dir

def take_screenshots(output_dir):
    """
    Takes screenshots of all HTML visualization files in the given directory.
    Captures SHAP, individual, and grouped visualizations.
    
    Args:
        output_dir (str): Directory containing HTML visualization files
    """
    # Setup Chrome in headless mode
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--log-level=3")  # Suppress console logs
    
    # Initialize browser
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )
    
    # Create screenshots directory if it doesn't exist
    screenshots_dir = os.path.join(output_dir, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)
    
    # Find all HTML files in the directory
    html_files = glob.glob(os.path.join(output_dir, "*.html"))
    
    for html_file in html_files:
        file_name = os.path.basename(html_file)
        
        # Extract fighter names and type from the filename based on new naming convention
        # Examples: ind_john_jane.html, grouped_john_jane.html, shap_john_jane.html
        match = re.search(r'(ind|grouped|shap)_([a-z]+)_([a-z]+)\.html', file_name, re.IGNORECASE)
        if not match:
            print(f"Skipping file with unrecognized pattern: {file_name}")
            continue
            
        viz_type, fighter1_first, fighter2_first = match.groups()
        
        # Create the screenshot filename
        screenshot_name = f"{viz_type}_{fighter1_first}_{fighter2_first}.png"
        screenshot_path = os.path.join(screenshots_dir, screenshot_name)
        
        # Load the HTML file
        file_url = f"file:///{os.path.abspath(html_file)}"
        driver.get(file_url)
        
        # Wait for the visualization to load
        time.sleep(2)
        
        # Take screenshot
        driver.save_screenshot(screenshot_path)
        print(f"Screenshot saved: {screenshot_path}")
        
        # Crop screenshot to remove excess whitespace - simplified approach
        try:
            img = Image.open(screenshot_path)
            img_data = np.array(img)
            
            # Convert to grayscale for content detection
            img_gray = np.mean(img_data, axis=2)
            height, width, _ = img_data.shape
            
            # Set threshold for what's considered "content" (non-white)
            # Lower values = more aggressive (detects lighter pixels as content)
            threshold = 240
            
            # Find rightmost non-white pixel
            right_boundary = 0
            # Scan from right to left
            for x in range(width - 1, width // 4, -1):  # Don't scan past 1/4 of the width
                column = img_gray[:, x]
                if np.min(column) < threshold:  # If any pixel in column is below threshold
                    right_boundary = x
                    break
            
            # Find bottommost non-white pixel
            bottom_boundary = 0
            # Scan from bottom to top
            for y in range(height - 1, height // 4, -1):  # Don't scan past 1/4 of the height
                row = img_gray[y, :]
                if np.min(row) < threshold:  # If any pixel in row is below threshold
                    bottom_boundary = y
                    break
            
            # Add padding
            right_padding = 60  # px
            bottom_padding = 40  # px
            
            # Calculate final boundaries with padding
            right_boundary = min(right_boundary + right_padding, width)
            bottom_boundary = min(bottom_boundary + bottom_padding, height)
            
            # Crop the image
            cropped = img.crop((0, 0, right_boundary, bottom_boundary))
            cropped.save(screenshot_path)
            print(f"  Cropped image from {width}x{height} to {right_boundary}x{bottom_boundary}")
        except Exception as e:
            print(f"Error cropping image {screenshot_path}: {e}")
    
    # Close the browser
    driver.quit()
    print(f"All screenshots saved to: {screenshots_dir}")

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        # Default directory to search for the most recent folder
        base_dir = str(picks_dir())
        
        # Find the most recent output directory
        dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))], 
                     key=lambda x: os.path.getmtime(os.path.join(base_dir, x)),
                     reverse=True)
        
        if dirs:
            output_dir = os.path.join(base_dir, dirs[0])
            print(f"Using most recent directory: {output_dir}")
        else:
            print("No output directories found in pics/picks folder.")
            sys.exit(1)
    
    take_screenshots(output_dir) 
