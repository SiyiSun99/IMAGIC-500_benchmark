import pandas as pd
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm

# Define cohorts, missing methods, and ratios
cohorts = ["C19"]
missing_methods = ["MCAR", "MAR", "MNAR"]
missing_ratios = [10, 20, 30, 40, 50]
SampleTime = 5
## NEED TO CHANGE THE PATH BASED ON YOUR COMPUTER ##
base_path = Path("/Users/siysun/Desktop/PhD/SynthCPHS_benchmark/data_stored")

## NEED TO CHANGE THE PATH BASED ON YOUR COMPUTER ##
# Output path for time recording
output_file = Path(
    "/Users/siysun/Desktop/PhD/SynthCPHS_benchmark/imputation_times_mean_mode_C19_synthetic.csv"
)
time_records = []

for cohort in cohorts:
    full_train_path = base_path / f"Completed_data/{cohort}/{cohort}_train.csv"
    full_train_data = pd.read_csv(full_train_path, header=0)
    full_test_path = base_path / f"Completed_data/{cohort}/{cohort}_test.csv"
    full_test_data = pd.read_csv(full_test_path, header=0)

    for miss_method in tqdm(missing_methods):
        for miss_ratio in tqdm(missing_ratios):
            save_dir_train = (
                f"data_mean/{cohort}/{cohort}_train/{miss_method}/miss{miss_ratio}"
            )
            train_output_dir = base_path / save_dir_train
            train_output_dir.mkdir(parents=True, exist_ok=True)

            save_dir_test = (
                f"data_mean/{cohort}/{cohort}_test/{miss_method}/miss{miss_ratio}"
            )
            test_output_dir = base_path / save_dir_test
            test_output_dir.mkdir(parents=True, exist_ok=True)

            imputation_times = []  # Store times for averaging

            for index_file in tqdm(range(SampleTime)):
                train_mask_path = (
                    base_path
                    / f"data_miss_mask/{cohort}/{cohort}_train/{miss_method}/miss{miss_ratio}/{index_file}.csv"
                )
                train_mask_n = pd.read_csv(train_mask_path, header=0).astype(bool)
                miss_train_data = full_train_data.mask(train_mask_n)

                test_mask_path = (
                    base_path
                    / f"data_miss_mask/{cohort}/{cohort}_test/{miss_method}/miss{miss_ratio}/{index_file}.csv"
                )
                test_mask_n = pd.read_csv(test_mask_path, header=0).astype(bool)
                miss_test_data = full_test_data.mask(test_mask_n)

                # Measure imputation time
                start_time = time.time()

                # categorical columns
                for col in miss_train_data[
                    [c for c in miss_train_data.columns if c[:3] == "cat"]
                ]:
                    miss_train_data[col].fillna(
                        miss_train_data[col].mode()[0], inplace=True
                    )

                # numeric columns
                for col in miss_train_data[
                    [c for c in miss_train_data.columns if c[:3] == "con"]
                ]:
                    miss_train_data[col].fillna(
                        round(miss_train_data[col].mean(), 1), inplace=True
                    )

                end_time = time.time()
                imputation_time = end_time - start_time
                imputation_times.append(imputation_time)

                # Save imputed data
                save_train_path = train_output_dir / f"{index_file}.csv"
                miss_train_data.to_csv(save_train_path, index=False)

                # categorical columns
                for col in miss_test_data[
                    [c for c in miss_test_data.columns if c[:3] == "cat"]
                ]:
                    miss_test_data[col].fillna(
                        miss_train_data[col].mode()[0], inplace=True
                    )

                # numeric columns
                for col in miss_test_data[
                    [c for c in miss_test_data.columns if c[:3] == "con"]
                ]:
                    miss_test_data[col].fillna(
                        round(miss_train_data[col].mean(), 1), inplace=True
                    )

                # Save imputed data
                save_test_path = test_output_dir / f"{index_file}.csv"
                miss_test_data.to_csv(save_test_path, index=False)

                # Compute average time
                avg_time = sum(imputation_times) / SampleTime

                # Store result
                time_records.append(
                    [cohort, "Mean/Mode", miss_method, miss_ratio, index_file, avg_time]
                )

                # Save results to CSV
                time_df = pd.DataFrame(
                    time_records,
                    columns=[
                        "Dataset",
                        "Method",
                        "Mechanism",
                        "MissingRatio",
                        "Index_file",
                        "AvgTime",
                    ],
                )
                time_df.to_csv(output_file, index=False)
                print(f"{miss_method}, {miss_ratio} Imputation times recorded.")
