library(mice)      # Required for missing data handling
library(parallel)  # Required for mclapply parallel processing
library(gdata)

# Load custom functions - adopted from the GitHub repo: 
# https://github.com/EagerSun/DL-vs-Stat_Impute
## NEED TO CHANGE THE PATH BASED ON YOUR COMPUTER
source('/Users/siysun/Desktop/PhD/SynthCPHS_benchmark/Generate_Missingness/amputation.R')

## NEED TO CHANGE THE PATH BASED ON YOUR COMPUTER
mainPath <- "/Users/siysun/Desktop/PhD/SynthCPHS_benchmark/data_stored/"
completeDataPath <- file.path(mainPath, "Completed_data")
missingDataPath <- file.path(mainPath, "data_miss")
cohort_miss_path <- file.path(missingDataPath, cohort)
train_missingDataPath <- file.path(cohort_miss_path, "C19_train")
test_missingDataPath <- file.path(cohort_miss_path, "C19_test")

# Create directories
dir.create(missingDataPath, showWarnings = FALSE)
dir.create(cohort_miss_path, showWarnings = FALSE)
dir.create(train_missingDataPath, showWarnings = FALSE)
dir.create(test_missingDataPath, showWarnings = FALSE)

# Missing data configuration
missingMechanism <- c("MCAR", "MAR", "MNAR")
missList <- c(10, 20, 30, 40, 50)
trainingSampleTime <- 1  # Number of samples for training

process_categorical_vars <- function(df) {
  # Get column names
  col_names <- names(df)
  
  # Find indices of categorical variables (starting with "cat_")
  cat_indices <- which(grepl("^cat_", col_names))
  
  if (length(cat_indices) > 0) {
    # Convert categorical columns to factors with explicit levels
    for (i in cat_indices) {
      # Remove any NA values before determining levels
      valid_values <- unique(df[[i]][!is.na(df[[i]])])
      df[[i]] <- factor(df[[i]], levels = valid_values)
    }
  } else {
    cat("No categorical variables (starting with 'cat_') found in the dataset\n")
  }
  
  return(df)
}

process_csv <- function(input_path, output_base_path_miss) {
  # Read the CSV file
  cat("Reading file:", input_path, "\n")
  full_df <- read.csv(input_path, header = 1)
  
  # Process categorical variables
  full_df <- process_categorical_vars(full_df)
  
  # Define columns to exclude from missing data generation
  # we don't generate missingness for the machine generated features
  exclude_cols <- c(1, 2, 3, 4, 6, 13)
  
  # Process for both missing data
  current_path <- output_base_path_miss
  current_samples <- trainingSampleTime
  
  # Create base output directory if it doesn't exist
  if (!dir.exists(current_path)) {
    dir.create(current_path, recursive = TRUE)
  }
  
  # Generate missing data for each mechanism
  for (method in missingMechanism) {
    method_path <- file.path(current_path, method)
    if (!dir.exists(method_path)) {
      dir.create(method_path)
    }
      
    # Generate for each missing percentage
    for (miss_rate in missList) {
      miss_path <- file.path(method_path, paste0("miss", miss_rate))
      if (!dir.exists(miss_path)) {
        dir.create(miss_path)
      }
    
      # Generate samples sequentially with error handling
      for (i in 1:current_samples) {
        tryCatch({
          file_name <- paste0(i-1, ".csv")
          file_path <- file.path(miss_path, file_name)
            
          # Check if file already exists
          if (!file.exists(file_path)) {
            # Create a copy of the dataframe for modification
            working_df <- full_df
              
            # Get indices of columns to include in missing data generation
            included_cols <- which(!seq_len(ncol(working_df)) %in% exclude_cols)
              
            # Store original column names for verification
            original_names <- names(working_df)
              
            # Generate missing values
            miss_result <- produce_NA(working_df[, included_cols, drop=FALSE], 
                                        mechanism = method, 
                                        perc.missing = miss_rate/100)
              
            # Verify the column order matches before assignment
            if (!identical(names(miss_result$data.incomp), names(working_df[, included_cols, drop=FALSE]))) {
              stop("Column order mismatch detected!")
            }
              
            # Assign missing values back to the working dataframe
            working_df[, included_cols] <- miss_result$data.incomp
              
            # Verify final column order matches original
            if (!identical(names(working_df), original_names)) {
              stop("Final column order does not match original!")
            }
              
            # Save the result
            write.csv(working_df, file_path, row.names = FALSE)
            cat(sprintf("Generated sample %d for %s, %s, miss%d\n", 
                        i-1, "miss", method, miss_rate))
          } else {
            cat(sprintf("File already exists: %s\n", file_path))
          }
        }, error = function(e) {
          cat(sprintf("Error generating sample %d: %s\n", i-1, e$message))
        })
      }
    }
  }
}

cohort <- "C19"  # Specify the cohort name

# Source data path
source_cohort_path <- file.path(completeDataPath, cohort)

# Process train cohort file
main_csv <- file.path(source_cohort_path, "C19_train.csv")
if (file.exists(main_csv)) {
  # Process the main file
  process_csv(main_csv, train_missingDataPath)
}

# # Process test cohort file
# main_csv <- file.path(source_cohort_path, "C19_test.csv")
# if (file.exists(main_csv)) {
#   # Process the main file
#   process_csv(main_csv, test_missingDataPath)
# }

