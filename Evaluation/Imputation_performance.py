#!/home/siyi.sun/miniconda3/bin python3
# -*- coding: UTF-8 -*-
"""
@Description :   Calculate imputation performance metrics for imputed datasets
@Author      :   siyi.sun
@Time        :   2025/03/10 23:09:49
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
from tqdm import tqdm
from sklearn.metrics import f1_score


def z_score_normalize(data, mean, std):
    """
    Apply z-score normalization using pre-computed mean and standard deviation
    """
    if std == 0:
        return np.zeros_like(data)
    return (data - mean) / std


cohorts = ["C19"]
missing_methods = ["MCAR", "MAR", "MNAR"]
missing_ratios = [10, 20, 30, 40, 50]
sample_time = 5
impute_methods = ["diffputer"]  # input the imputation methods you want to evaluate
mode = "train"  # "train" or "test"
base_path = Path("/Users/siysun/Desktop/PhD/SynthCPHS_benchmark/data_stored")

results = []

for method in impute_methods:
    for cohort in cohorts:
        # Load full dataset
        full_data_path = base_path / f"Completed_data/{cohort}/{cohort}_{mode}.csv"
        full_data = pd.read_csv(full_data_path)

        # Get continuous and categorical columns
        cont_cols = [col for col in full_data.columns if col.startswith("con_")]
        cat_cols = [col for col in full_data.columns if col.startswith("cat_")]

        # Calculate mean and standard deviation for each continuous column in full dataset
        cont_means = {col: full_data[col].mean() for col in cont_cols}
        cont_stds = {col: full_data[col].std() for col in cont_cols}

        for miss_method in tqdm(missing_methods):
            for miss_ratio in tqdm(missing_ratios):
                cont_rmse_list = []
                cat_acc_list = []
                cat_f1_score_list = []

                # Process each sample for each scenario
                for sample_idx in range(sample_time):
                    # Load missing mask
                    mask_path = (
                        Path(base_path)
                        / f"data_miss_mask/{cohort}/{cohort}_{mode}/{miss_method}/miss{miss_ratio}/{sample_idx}.csv"
                    )
                    mask_data = pd.read_csv(mask_path)

                    # Load imputed data
                    imputed_path = (
                        Path(base_path)
                        / f"data_{method}/{cohort}/{cohort}_{mode}/{miss_method}/miss{miss_ratio}/{sample_idx}.csv"
                    )
                    imputed_data = pd.read_csv(imputed_path)

                    # Calculate RMSE for continuous variables
                    if cont_cols:
                        cont_rmse = 0
                        total_missing = 0

                        for col in cont_cols:
                            # Get indices where values were missing
                            missing_idx = mask_data[col] == 1
                            n_missing = missing_idx.sum()

                            if n_missing > 0:
                                # Normalize both full and imputed data using full data's min-max
                                full_normalized = z_score_normalize(
                                    full_data.loc[missing_idx, col],
                                    cont_means[col],
                                    cont_stds[col],
                                )
                                imputed_normalized = z_score_normalize(
                                    imputed_data.loc[missing_idx, col],
                                    cont_means[col],
                                    cont_stds[col],
                                )
                                # Calculate squared errors using normalized values
                                squared_errors = (
                                    full_normalized - imputed_normalized
                                ) ** 2
                                cont_rmse += squared_errors.sum()
                                total_missing += n_missing
                                print(
                                    f"{col}: square error = {squared_errors.sum()}, missing_number = {n_missing}"
                                )

                        if total_missing > 0:
                            cont_rmse = np.sqrt(cont_rmse / total_missing)
                            cont_rmse_list.append(cont_rmse)
                            print(total_missing)

                    # Calculate accuracy and ROC AUC for categorical variables
                    if cat_cols:
                        cat_acc = 0
                        total_missing = 0
                        cat_f1_score = 0
                        total_cols_with_f1_score = 0

                        for col in cat_cols:
                            missing_idx = mask_data[col] == 1
                            n_missing = missing_idx.sum()

                            if n_missing > 0:
                                # Calculate accuracy for missing values
                                correct = (
                                    full_data.loc[missing_idx, col]
                                    == imputed_data.loc[missing_idx, col]
                                ).sum()
                                cat_acc += correct
                                total_missing += n_missing

                                # Calculate F1 score
                                # Get true values and predicted values
                                y_true = full_data.loc[missing_idx, col].values
                                y_pred = imputed_data.loc[missing_idx, col].values

                                col_f1_score = f1_score(y_true, y_pred, average="macro")
                                cat_f1_score += col_f1_score
                                total_cols_with_f1_score += 1
                                print(f"{col}: ROC AUC = {col_f1_score}")

                        if total_missing > 0:
                            cat_acc = cat_acc / total_missing
                            cat_acc_list.append(cat_acc)

                        if total_cols_with_f1_score > 0:
                            cat_f1_score = cat_f1_score / total_cols_with_f1_score
                            cat_f1_score_list.append(cat_f1_score)

                # Calculate mean and variance of metrics
                cont_rmse_mean = np.mean(cont_rmse_list) if cont_rmse_list else np.nan
                cont_rmse_var = np.var(cont_rmse_list) if cont_rmse_list else np.nan
                cat_acc_mean = np.mean(cat_acc_list) if cat_acc_list else np.nan
                cat_acc_var = np.var(cat_acc_list) if cat_acc_list else np.nan
                cat_f1_score_mean = (
                    np.mean(cat_f1_score_list) if cat_f1_score_list else np.nan
                )
                cat_f1_score_var = (
                    np.var(cat_f1_score_list) if cat_f1_score_list else np.nan
                )

                # Add results to list
                results.append(
                    {
                        "impute_method": method,
                        "cohort": cohort,
                        "missing_mechanism": miss_method,
                        "missing_ratio": miss_ratio,
                        "mean_continuous_rmse": cont_rmse_mean,
                        "var_continuous_rmse": cont_rmse_var,
                        "mean_categorical_accuracy": cat_acc_mean,
                        "var_categorical_accuracy": cat_acc_var,
                        "mean_categorical_f1_score": cat_f1_score_mean,
                        "var_categorical_f1_score": cat_f1_score_var,
                    }
                )

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV
output_path = f"/Users/siysun/Desktop/PhD/SynthCPHS_benchmark/imputation_metrics_results_C19_Synthetic_{method}_{mode}.csv"
results_df.to_csv(output_path, index=False)
