#!/home/siyi.sun/miniconda3/bin python3
# -*- coding: UTF-8 -*-
"""
@Description :   Add cat_hid column to each TabCSDI imputed CSV file in the directory
@Author      :   siyi.sun
@Time        :   2025/04/12 04:19:32
"""

import pandas as pd
import os
from pathlib import Path


def process_file(file_path: str, relative_path: str, cat_hid):
    """Process a single CSV file and add the cat_hid column."""
    # Read the data file
    data = pd.read_csv(file_path, header=0)
    data = pd.concat([cat_hid, data], axis=1)
    data.to_csv(file_path, index=False)


def process_directory(current_dir: str, base_dir: str, cat_hid):
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
                process_file(file_path, relative_path, cat_hid)


cohort = "SynthSurvey"
# Define the base path
base_path = Path("../data_stored")

# Process the train data
train_path = base_path / f"Completed_data/{cohort}/{cohort}_train.csv"
train_data = pd.read_csv(train_path)
tabcsdi_train_path = base_path / "data_tabcsdi/SynthSurvey/SynthSurvey_train"
process_directory(tabcsdi_train_path, tabcsdi_train_path, train_data["cat_hid"])

# Process the test data
test_path = base_path / f"Completed_data/{cohort}/{cohort}_test.csv"
test_data = pd.read_csv(test_path)
tabcsdi_test_path = base_path / "data_tabcsdi/SynthSurvey/SynthSurvey_test"
process_directory(tabcsdi_test_path, tabcsdi_test_path, test_data["cat_hid"])
