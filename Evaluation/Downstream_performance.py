#!/home/siyi.sun/miniconda3/bin python3
# -*- coding: UTF-8 -*-
"""
@Description :   Calculate downstream performance metrics for imputed datasets
@Author      :   siyi.sun
@Time        :   2025/03/21 02:23:26
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from category_encoders.leave_one_out import LeaveOneOutEncoder  # Explicit import
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

X = pd.read_csv(f"../data_stored/Completed_data/SynthSurvey/SynthSurvey_train.csv")
# "cat_hid" is not used as feature for downstream task
X.drop(columns = "cat_hid", inplace = True)

dict_bin = {1:0,2:1} 
# handle binary data;
X["cat_urbrur"] = X["cat_urbrur"].map(dict_bin)
X["cat_sex"] = X["cat_sex"].map(dict_bin)

################################# Train ###############################

# handling categorical data -> three ways:
# 1. if it's binary string -> map to 0,1 (the dominant category as 0)
# 2. if it's binary number 0,1 -> keep it
# 3. if it's multiclass (>4) -> Leave one out encoder
categorical_features = ['cat_geo1', 'cat_geo2', 'cat_statocc',
                        'cat_relation', 'cat_marstat', 'cat_religion',
                        'cat_act_status', 'cat_occupation']
y = pd.read_csv(f"../data_stored/Completed_data/SynthSurvey/SynthSurvey_train_y.csv")["cat_educ_attain"]

############################### Test True ###########################

X_test = pd.read_csv(f"../data_stored/Completed_data/SynthSurvey/SynthSurvey_test.csv")
y_test = pd.read_csv(f"../data_stored/Completed_data/SynthSurvey/SynthSurvey_test_y.csv")["cat_educ_attain"]

# Print class distribution
print("Class distribution in training set:")
print(y.value_counts(normalize=True))

# Create preprocessing steps
categorical_transformer = LeaveOneOutEncoder(
    cols=categorical_features, random_state=42, sigma=0.05
)

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        (
            "binary_cat",
            "passthrough",
            [
                col
                for col in X.columns
                if col[:3] == "cat" and col not in categorical_features
            ],
        ),
        ("num", StandardScaler(), [col for col in X.columns if col[:3] == "con"]),
    ]
)

# Option 2: Alternative approach - exclude very rare classes from SMOTE
# Get classes with at least 2 samples (minimum needed for k_neighbors=1)
classes_to_oversample = y.value_counts()[y.value_counts() >= 2].index.tolist()
sampling_strategy = {cls: max(y.value_counts()) for cls in classes_to_oversample if cls != y.value_counts().idxmax()}
smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy, k_neighbors=1)

# Create imbalanced pipeline with SMOTE
pipeline = ImbPipeline(
    [
        ("preprocessor", preprocessor),
        ("smote",smote),  # Adjust sampling_strategy as needed
        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                class_weight="balanced",  # Use balanced class weights
                n_estimators=200,  # Increase number of trees
                max_depth=None,  # Allow deep trees to capture rare patterns
                min_samples_leaf=1,  # Allow leaf nodes to be smaller
            ),
        ),
    ]
)

# Fit the pipeline
pipeline.fit(X, y)

# ################# Test on Full test set (serve as baseline) #################
# # Make predictions
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)

# Print classification report with detailed metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# For multiclass, use OVR (One-vs-Rest) approach
print("\nMulticlass ROC AUC Scores:")
    
# Calculate ROC AUC for each class
for i, class_name in enumerate(np.unique(y_test)):
    # If your classes are not 0,1,2,... but have actual meaningful values
    class_index = np.where(pipeline.classes_ == class_name)[0][0]
    class_proba = y_pred_proba[:, class_index]

    # Create binary labels for this class (one-vs-rest)
    y_test_binary = (y_test == class_name).astype(int)

    try:
        class_auc = roc_auc_score(y_test_binary, class_proba)
        print(f"Class {class_name} ROC AUC: {class_auc:.4f}")
    except Exception as e:
        print(f"Could not calculate AUC for class {class_name}: {e}")

weighted_auc = roc_auc_score(
    y_test, 
    y_pred_proba,
    multi_class='ovr',
    average='weighted'
)
print(f"\nWeighted Average ROC AUC: {weighted_auc:.4f}")
    
print(f"Average Precision Score: {average_precision_score(y_test, y_pred_proba):.4f}")

# Feature importance
feature_importance = pd.DataFrame(
    pipeline.named_steps["classifier"].feature_importances_,
    index=X.columns,
    columns=["importance"],
).sort_values("importance", ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

########################### Test on Imputed Data ###########################
# General Parameters
miss_mechanism = [
    "MCAR",
    "MAR",
    "MNAR",
]  # input the missing mechanism u want to evaluate
miss_ratio = [10, 20, 30, 40, 50]  # input the missing ratio u want to evaluate
cohort = "SynthSurvey"
methods_folder = [
    "data_mean",
    "data_mice",
    "data_missforest",
    "data_gain",
    "data_dsan",
    "data_dsn",
    "data_softimpute",
    "data_diffputer",
    "data_MOT",
    "data_remasker",
    "data_miwae",
    "data_hyperimpute",
    "data_miracle",
]  # input the imputation methods u want to evaluate
# Note: if you evaluate part of the methods, you need to combinre the result manually
test_number = 5  # sample times
base_path = Path("/Users/siysun/Desktop/NeurIPS25/data_stored")
roc = []
acc = []

# original dataset
test_data = pd.read_csv(f"../data_stored/Completed_data/{cohort}/{cohort}_test.csv")

for mechanism in tqdm(miss_mechanism):
    for ratio in tqdm(miss_ratio):
        roc_methods = [mechanism, ratio]
        acc_methods = [mechanism, ratio]
        for folder in tqdm(methods_folder):
            score_list_roc = []
            score_list_acc = []
            for n in tqdm(range(test_number)):
                # Load missing mask
                mask_path = (
                    Path(base_path)
                    / f"data_miss_mask/{cohort}/{cohort}_test/{mechanism}/miss{ratio}/{n}.csv"
                )
                mask_data = pd.read_csv(mask_path)

                # Load imputed data
                imputed_path = (
                    Path(base_path)
                    / f"{folder}/{cohort}/{cohort}_test/{mechanism}/miss{ratio}/{n}.csv"
                )
                imputed_data = pd.read_csv(imputed_path)

                # Create a copy of the original dataset to avoid modifying it
                result_dataset = test_data.copy()
                result_dataset = result_dataset.mask(
                    mask_data.astype(bool), imputed_data
                )
                imputed_data = result_dataset

                # Make predictions
                y_pred = pipeline.predict(imputed_data)
                y_pred_proba = pipeline.predict_proba(imputed_data)[:, 1]

                score_list_roc.append(roc_auc_score(y_test, y_pred_proba))
                score_list_acc.append(accuracy_score(y_test, y_pred))
            roc_methods.append(
                f"{round(np.mean(score_list_roc),3)}\u00b1{round(np.std(score_list_roc),3)}"
            )
            acc_methods.append(
                f"{round(np.mean(score_list_acc),3)}\u00b1{round(np.std(score_list_acc),3)}"
            )
        roc.append(roc_methods)
        acc.append(acc_methods)

# Save results to CSV files
pd.DataFrame(
    roc, columns=["missing_machenism", "missing_ratio"] + methods_folder
).to_csv(f"/Users/siysun/Desktop/NeurIPS25/Downstream_temp_ROC.csv")
pd.DataFrame(
    acc, columns=["missing_machenism", "missing_ratio"] + methods_folder
).to_csv(f"/Users/siysun/Desktop/NeurIPS25/Downstream_temp_ACC.csv")

######################## (Optional) Calculate the performance degradation ########################
# def percent_roc(s):
#     mean_v = float(s.split('\u00B1')[0])
#     std_v = float(s.split('\u00B1')[1])
#     mean_p = round((mean_v-0.7378)*100/0.7378,1) # replace 0.7378 with the baseline ROC-AUC score
#     std_p = round(std_v*100/0.7378,1) # replace 0.7378 with the baseline ROC-AUC score
#     new_s = str(mean_p)+'%\u00B1'+str(std_p)+'%'
#     return new_s

# def percent_acc(s):
#     mean_v = float(s.split('\u00B1')[0])
#     std_v = float(s.split('\u00B1')[1])
#     mean_p = round((mean_v-0.9954)*100/0.9954,1)  # replace 0.9954 with the baseline accuracy score
#     std_p = round(std_v*100/0.9954,1)  # replace 0.9954 with the baseline accuracy score
#     new_s = str(mean_p)+'%\u00B1'+str(std_p)+'%'
#     return new_s

# # suppose you combine the results of all methods (axis = 1) into this csv file (e.g. ROC-AUC score)
# results = pd.read_csv('../C19_synthetic_ROC_downstream.csv', index_col=None,header = 0)
# results_new = results.iloc[:,2:].applymap(percent_roc)  # YOU CAN REPLACE THE FUNCTION NAME WITH percent_acc
# results_new.to_csv('downstream_performance_ROC.csv',index = False)
