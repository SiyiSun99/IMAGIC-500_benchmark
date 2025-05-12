#!/home/siyi.sun/miniconda3/bin python3
# -*- coding: UTF-8 -*-
"""
@Description :   map missingness in each file to mask (0-not missing;1-missing)
@Author      :   siyi.sun
@Time        :   2025/03/03 01:45:24
"""

import os
import pandas as pd
import numpy as np


def create_missing_masks(source_dir: str, target_dir: str):
    """
    Create missing value mask files from CSV files containing missing data.

    Args:
        source_dir (str): Path to source directory containing CSV files with missing values
        target_dir (str): Path to target directory where mask files will be stored
    """
    # Create the target base directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    def process_file(file_path: str, relative_path: str):
        """Process a single CSV file and create its corresponding mask file."""
        try:
            # Read the data file
            data = pd.read_csv(file_path, header=0)

            # Create missing value mask (1 for missing, 0 for non-missing)
            mask = data.isna().astype(int)

            # Create the corresponding directory structure in target_dir
            target_file_dir = os.path.dirname(os.path.join(target_dir, relative_path))
            if not os.path.exists(target_file_dir):
                os.makedirs(target_file_dir)

            # Save the mask
            target_file_path = os.path.join(target_dir, relative_path)
            mask.to_csv(target_file_path, index=False)

            print(f"Processed: {relative_path}")

        except Exception as e:
            print(f"Error processing {relative_path}: {str(e)}")

    def process_directory(current_dir: str, base_dir: str):
        """Recursively process directories and their contents."""
        # List all files and directories
        for root, _, files in os.walk(current_dir):
            for file in files:
                if file.endswith(".csv"):
                    # Get full file path
                    file_path = os.path.join(root, file)
                    # Get relative path
                    relative_path = os.path.relpath(file_path, base_dir)
                    # Process the file
                    process_file(file_path, relative_path)

    try:
        process_directory(source_dir, source_dir)
        print("Missing value mask creation completed successfully!")
    except Exception as e:
        print(f"Error during processing: {str(e)}")


if __name__ == "__main__":
    ## NEED TO CHANGE THE PATH BASED ON YOUR COMPUTER ##
    source_dir = (
        "/Users/siysun/Desktop/NeurIPS25/data_stored/data_miss/SynthSurvey"
    )
    target_dir = (
        "/Users/siysun/Desktop/NeurIPS25/data_stored/data_miss_mask/SynthSurvey"
    )
    create_missing_masks(source_dir, target_dir)
