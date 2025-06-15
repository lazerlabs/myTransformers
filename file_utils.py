import os
import warnings
from typing import List


def find_csv_files(path: str) -> List[str]:
    """
    Recursively find all CSV files in the given path.
    
    Args:
        path (str): Path to a file or directory
        
    Returns:
        List[str]: List of CSV file paths
    """
    if os.path.isfile(path):
        # If it's a file, check if it's a CSV
        if path.lower().endswith('.csv'):
            return [path]
        else:
            warnings.warn(f"File {path} is not a CSV file. Skipping.")
            return []
    elif os.path.isdir(path):
        # If it's a directory, recursively find all CSV files
        csv_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.lower().endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        return sorted(csv_files)  # Sort for consistent ordering
    else:
        warnings.warn(f"Path {path} does not exist. Skipping.")
        return [] 
