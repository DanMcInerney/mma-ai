"""Base class for data loaders."""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple
import pandas as pd


def select_duplicate_fighter_id(fighter_data: pd.DataFrame, fighter_name: str) -> Any | None:
    """
    Select a fighter_id deterministically when a fighter name maps to multiple IDs.

    Prediction runs from the web UI cannot safely pause for stdin. The most useful
    default for upcoming fights is the ID with the most recent event_date; row
    count and ID text provide stable tie breakers.
    """
    if "fighter_id" not in fighter_data.columns:
        return None

    valid_data = fighter_data[fighter_data["fighter_id"].notna()].copy()
    if valid_data.empty:
        return None

    candidates: list[dict[str, Any]] = []
    for fighter_id, group in valid_data.groupby("fighter_id", sort=False):
        if "event_date" in group.columns:
            dates = pd.to_datetime(group["event_date"], errors="coerce")
            latest = dates.max()
        else:
            latest = pd.NaT
        latest_score = int(latest.value) if pd.notna(latest) else -(10**30)
        latest_label = latest.strftime("%Y-%m-%d") if pd.notna(latest) else "unknown date"
        candidates.append(
            {
                "latest_score": latest_score,
                "rows": len(group),
                "id_label": str(fighter_id),
                "fighter_id": fighter_id,
                "latest_label": latest_label,
            }
        )

    candidates.sort(
        key=lambda candidate: (candidate["latest_score"], candidate["rows"], candidate["id_label"]),
        reverse=True,
    )
    selected = candidates[0]
    print(
        f"Auto-selected fighter_id {selected['fighter_id']} for {fighter_name} "
        f"(latest fight {selected['latest_label']}, {selected['rows']} rows)."
    )
    return selected["fighter_id"]


class DataLoader(ABC):
    """Abstract base class for loading fighter data from CSV."""
    
    @abstractmethod
    def load_fighter_data(self, fighter_name: str) -> pd.DataFrame:
        """
        Load data for a specific fighter.
        
        Args:
            fighter_name: Name of the fighter
            
        Returns:
            DataFrame with fighter's historical data
        """
        pass
    
    @abstractmethod
    def filter_fighters(self, fight_list: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """
        Filter fight list to remove fights with insufficient data.
        
        Args:
            fight_list: List of tuples (fight_date, fighter1_name, fighter2_name)
            
        Returns:
            Filtered fight list
        """
        pass
    
    def handle_duplicate_ids(self, fighter_data: pd.DataFrame, fighter_name: str) -> pd.DataFrame:
        """
        Handle cases where a fighter has multiple fighter_ids.
        
        Args:
            fighter_data: DataFrame with potential duplicate IDs
            fighter_name: Name of the fighter
            
        Returns:
            DataFrame with single fighter_id
        """
        if 'fighter_id' not in fighter_data.columns:
            return fighter_data
            
        unique_ids = fighter_data['fighter_id'].dropna().unique()
        if len(unique_ids) > 1:
            print(f"Warning: Multiple fighter_ids found for {fighter_name}: {unique_ids}")
            
            print(f"--- Information to help select the correct ID for {fighter_name} ---")
            for fid in unique_ids:
                fighter_history_for_id = fighter_data[fighter_data['fighter_id'] == fid].copy()
                if not fighter_history_for_id.empty:
                    if 'event_date' in fighter_history_for_id.columns:
                        fighter_history_for_id['event_date'] = pd.to_datetime(fighter_history_for_id['event_date'])
                        fighter_history_for_id = fighter_history_for_id.sort_values(by='event_date', ascending=False)
                        if not fighter_history_for_id.empty:
                            last_recorded_fight = fighter_history_for_id.iloc[0]
                            date_of_last_fight = last_recorded_fight['event_date'].strftime('%Y-%m-%d')
                            print(f"  ID {fid}: Last fought on {date_of_last_fight}.")
                        else:
                            print(f"  ID {fid}: No fights found for this ID after sorting.")
                    else:
                        print(f"  ID {fid}: No event_date column found.")
                else:
                    print(f"  ID {fid}: No fight history found in the dataset for this specific ID.")
            print("--- End of disambiguation information ---")
            
            correct_fighter_id = select_duplicate_fighter_id(fighter_data, fighter_name)
            if correct_fighter_id is None:
                print(f"Error: Could not select a fighter_id for {fighter_name}.")
                return pd.DataFrame()
            fighter_data = fighter_data[fighter_data['fighter_id'] == correct_fighter_id].copy()
        elif len(unique_ids) == 0:
            print(f"Warning: No fighter_id found for {fighter_name}.")
            return pd.DataFrame()
            
        return fighter_data

