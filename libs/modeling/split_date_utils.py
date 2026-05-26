import os
import pandas as pd
from pathlib import Path

def get_data_dir():
    """Get the data directory path relative to this file."""
    # Go up to project root from libs/modeling/
    project_root = Path(__file__).parent.parent.parent
    return project_root / "data"

def write_test_start_date(test_start_date, path):
    """Write the test data start date to a file in the data directory.
    
    Args:
        test_start_date: pandas Timestamp or string representing the test start date
    """
    if not path:
        data_dir = get_data_dir()
        data_dir.mkdir(exist_ok=True)
        path = data_dir / "test_start_date.txt"
    
    # Convert to string if it's a Timestamp
    if isinstance(test_start_date, pd.Timestamp):
        date_str = test_start_date.strftime('%Y-%m-%d')
    else:
        date_str = str(test_start_date)
    
    #test_date_file = data_dir / "test_start_date.txt"
    
    with open(path, 'w') as f:
        f.write(date_str)
    
    print(f"Test start date '{date_str}' written to {path}")

def read_test_start_date(path):
    """Read the test data start date from the file in the data directory.
    
    Returns:
        str: The test start date as a string (YYYY-MM-DD format)
        
    Raises:
        FileNotFoundError: If the test start date file doesn't exist
        ValueError: If the file is empty or contains invalid data
    """
    if not path:
        data_dir = get_data_dir()
        data_dir.mkdir(exist_ok=True)
        path = data_dir / "test_start_date.txt"
    else:
        # Convert string path to Path object if needed
        if isinstance(path, str):
            path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Test start date file not found at {path}. "
                              "Make sure train.py has been run to generate this file.")
    
    with open(path, 'r') as f:
        date_str = f.read().strip()
    
    if not date_str:
        raise ValueError(f"Test start date file {path} is empty.")
    
    # Validate the date format
    try:
        pd.to_datetime(date_str)
    except Exception as e:
        raise ValueError(f"Invalid date format in {path}: '{date_str}'. "
                        f"Expected YYYY-MM-DD format. Error: {e}")
    
    return date_str

def get_test_start_date_as_timestamp(path):
    """Read the test start date and return it as a pandas Timestamp.
    
    Returns:
        pd.Timestamp: The test start date as a pandas Timestamp
    """
    date_str = read_test_start_date(path)
    return pd.to_datetime(date_str) 

def write_train_end_date(train_end_date):
    """Write the train data end date to a file in the data directory.
    
    Args:
        train_end_date: pandas Timestamp or string representing the train end date
    """
    data_dir = get_data_dir()
    data_dir.mkdir(exist_ok=True)
    
    # Convert to string if it's a Timestamp
    if isinstance(train_end_date, pd.Timestamp):
        date_str = train_end_date.strftime('%Y-%m-%d')
    else:
        date_str = str(train_end_date)
    
    train_end_date_file = data_dir / "train_end_date.txt"
    
    with open(train_end_date_file, 'w') as f:
        f.write(date_str)
    
    print(f"Train end date '{date_str}' written to {train_end_date_file}") 

def read_train_end_date():
    """Read the train data end date from the file in the data directory.
    
    Returns:
        str: The train end date as a string (YYYY-MM-DD format)
        
    Raises:
        FileNotFoundError: If the train end date file doesn't exist
        ValueError: If the file is empty or contains invalid data
    """
    data_dir = get_data_dir()
    train_end_date_file = data_dir / "train_end_date.txt"
    
    if not train_end_date_file.exists():
        raise FileNotFoundError(f"Train end date file not found at {train_end_date_file}. "
                              "Make sure train.py has been run to generate this file.")
    
    with open(train_end_date_file, 'r') as f:
        date_str = f.read().strip()
    
    if not date_str:
        raise ValueError(f"Train end date file {train_end_date_file} is empty.")
    
    # Validate the date format
    try:
        pd.to_datetime(date_str)
    except Exception as e:
        raise ValueError(f"Invalid date format in {train_end_date_file}: '{date_str}'. "
                        f"Expected YYYY-MM-DD format. Error: {e}")
    
    return date_str

def get_train_end_date_as_timestamp():
    """Read the train end date and return it as a pandas Timestamp.
    
    Returns:
        pd.Timestamp: The train end date as a pandas Timestamp
    """
    date_str = read_train_end_date()
    return pd.to_datetime(date_str) 