#!/usr/bin/env python3
"""
Script to run the full pipeline test independently.
This test creates dummy data and runs it through the actual main.py pipeline logic.
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import pytest

if __name__ == "__main__":
    # Run the pipeline test with verbose output
    test_file = os.path.join(os.path.dirname(__file__), "test_full_pipeline.py")
    
    print("=" * 80)
    print("RUNNING FULL PIPELINE TEST")
    print("=" * 80)
    print(f"Test file: {test_file}")
    print()
    
    # Run with verbose output and stop on first failure
    exit_code = pytest.main([
        test_file,
        "-v",           # Verbose output
        "-s",           # Don't capture output (show prints)
        "--tb=short",   # Short traceback format
        "-x"            # Stop on first failure
    ])
    
    print()
    if exit_code == 0:
        print("✅ All pipeline tests passed!")
    else:
        print("❌ Pipeline tests failed!")
    
    sys.exit(exit_code)
