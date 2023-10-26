# clear the env
rm(list = ls())
gc()

################################################################# IMPORT PACKAGE
library(readr)
library(dplyr)
library(janitor)
library(naniar)
library(dplyr)
library(tidyr)
library(caret)
library(tictoc)
library(ranger)
library(DMwR2)
library(doParallel)
library(foreach)
library(magrittr)
library(styler)
library(xgboost)
library(missranger)

################################################### preprocessing & manipulation
tic()
# Import dataset
df <- suppressMessages(read_csv('combined.csv'))

# Transform col names to clean names
df <- clean_names(df)
df %>%
  glimpse # take a glimpse of dataframe

# sum up the total null values in dataframe
df %>%
  is.na %>%
  sum

# Visualization of na data
miss_var_table(df)
vis_miss(df, warn_large_data = FALSE)
gg_miss_var(df)

# set up missing test function
missingness_test <- function(df, col_name) {
  df <- df %>% # mutate a column to tell if data is missing or not
    mutate(!!paste0("missing_", col_name) := is.na(!!sym(col_name)))
  
  missing_com <-
    df %>% # keep rows which values is commercial and pull from missing cols
    filter(residential_commercial == "Commercial") %>%
    pull(!!paste0("missing_", col_name))
  
  missing_res <-
    df %>% # keeps rows which values is residential and pull from missing cols
    filter(residential_commercial == "Residential") %>%
    pull(!!paste0("missing_", col_name))
  
  t.test(missing_com, missing_res) # conduct t test
}

# Test missingness for each column
columns_to_test <-
  # are the columns with missing values needs to be tested
  c(
    "eviction_apartment_number",
    "bin",
    "bbl",
    "nta",
    "longitude",
    "latitude",
    "council_district",
    "community_board",
    "census_tract"
  )

results <-
  lapply(columns_to_test, function(col_name) {
    # apply function to the list
    missingness_test(df, col_name)
  })

# Print test results
results

# Function to count unique values in columns
count_unique <- function(df, columns) {
  df %>%
    select(all_of(columns)) %>%
    summarize_all( ~ length(unique(.)))
}

# remove unused dataframe and free the memory
rm(df)
gc()

# Import original data
df_ori <-
  suppressMessages(read.csv('combined.csv', strip.white = TRUE))

# Clean column names
df_ori <- clean_names(df_ori)

# Glimpse the dataframe
df_ori %>%
  glimpse()

# Check the unique values in each missing column
missing_columns <- c(
  "eviction_apartment_number",
  "latitude",
  "longitude",
  "community_board",
  "council_district",
  "census_tract",
  "bin",
  "bbl",
  "nta"
)

# count unique values in missing_columns
count_unique(df_ori, missing_columns)

# Function to separate and encode related columns
preprocessing <- function(df) {
  df %>%
    separate(
      # separate date into year, month and day
      executed_date,
      into = c("year", "month", "day"),
      sep = "-",
      convert = TRUE
    ) %>%
    mutate(
      # mutate several columns and using mathmatics transformation to time related variables
      year = as.numeric(year),
      month = as.numeric(month),
      day = as.numeric(day),
      month_sin = sin(month * 2 * pi / 12),
      month_cos = cos(month * 2 * pi / 12),
      day_sin = sin(day * 2 * pi / 31),
      day_cos = cos(day * 2 * pi / 31),
      encoded_year = as.numeric(factor(
        # consider years as ordinal variables and sort order
        year, ordered = TRUE, levels = sort(unique(year))
      ))
    )
}

# set up function to visualize missing data and for future use
visualize_missing_data <- function(data) {
  # visualize dataframe
  vis_miss(data, warn_large_data = FALSE)
}

# Process the original dataframe
df_processed <- df_ori %>%
  preprocessing() %>%
  unite("marshal_full_name",
        marshal_first_name,
        marshal_last_name,
        sep = " ") %>%
  select(
    -c(
      court_index_number,
      eviction_address,
      eviction_apartment_number,
      year,
      month,
      day,
      nta
    )
  )

# Glimpse the dataframe
df_ori %>%
  glimpse()

# Split the data into train, validation, and test sets
set.seed(1234)
train_ind <-
  createDataPartition(df_processed$economic_need_index,
                      p = 0.7,
                      list = FALSE)
train <- df_processed[train_ind,]
remain <- df_processed[-train_ind,]
val_ind <-
  createDataPartition(remain$economic_need_index, p = 0.5, list = FALSE)
val <- remain[val_ind,]
test <- remain[-val_ind,]

# Visualize missing data
visualize_missing_data(train)
visualize_missing_data(test)
visualize_missing_data(val)

############################################################ feature engineering
# Define a function for encoding data
encoding <- function(data) {
  data_enc <- data %>% group_by(marshal_full_name) %>%
    mutate(name_count = n()) %>%
    ungroup() %>% select(-marshal_full_name)
  
  template <- # create dummy variables template
    dummyVars(~ borough + residential_commercial, data = data)
  encoded <- predict(template, newdata = data_enc)
  encoded_df <- as.data.frame(encoded)
  
  cols <- c("borough", "residential_commercial")
  data_enc <-
    cbind(data_enc[, !names(data_enc) %in% cols], encoded_df)
  data_enc <- data_enc %>% clean_names()
  
  return(data_enc)
}

# Encode train, test, and validation data
train_enc <- encoding(train)
test_enc <- encoding(test)
val_enc <- encoding(val)

# Separate target and predictors in train set
y_train <- train_enc$economic_need_index
x_train <- train_enc %>% select(-economic_need_index)

# Separate target and predictors in test set
y_test <- test_enc$economic_need_index
x_test <- test_enc %>% select(-economic_need_index)

# Separate target and predictors in validation set
y_val <- val_enc$economic_need_index
x_val <- val_enc %>% select(-economic_need_index)

# take a glimpse of x_train, x_Test and x_val
x_train %>% glimpse
x_test %>% glimpse
x_val %>% glimpse

# remove unused dataframe and free the memory
rm(df_ori, df_processed, remain, results, test, train, val)
gc()

# results comparasion of varies MAR imputation methods
# create  function to impute missing values with the median
# Impute missing values in the training, validation, and test datasets
# initialize list of imputation method
########## median imputation method
# Function for median imputation
impute_median <- function(data, medians = NULL) {
  if (is.null(medians)) {
    medians <-
      apply(data, 2, function(x)
        # if median nulls, calculate medians of each columns
        median(x, na.rm = TRUE))
  }
  
  data_imp_median <- data.frame(Map(function(x, med)
    ifelse(is.na(x), med, x), data, medians))
  
  return(list(data = data_imp_median, medians = medians))
}

# Function for ranger model training and prediction
train_predict_ranger <- function(x_train, y_train, x_pred) {
  model <-
    ranger(y_train ~ ., data = x_train, importance = "permutation")
  predictions <- predict(model, data = x_pred)$predictions
  return(as.numeric(predictions))
}

# Impute median values
train_imp_result <- impute_median(x_train)
train_imp_median <- train_imp_result$data
medians_train <- train_imp_result$medians

val_imp_result <- impute_median(x_val, medians_train)
val_imp_median <- val_imp_result$data

test_imp_result <- impute_median(x_test, medians_train)
test_imp_median <- test_imp_result$data

# Train and predict with ranger model
val_pred_median <-
  train_predict_ranger(train_imp_median, y_train, val_imp_median)
test_pred_median <-
  train_predict_ranger(train_imp_median, y_train, test_imp_median)

# Calculate RMSE
rmse_val_median <- RMSE(val_pred_median, y_val)
rmse_test_median <- RMSE(test_pred_median, y_test)

# Print RMSE
cat("Validation RMSE (using median):", rmse_val_median, "\n")
cat("Test RMSE (using median):", rmse_test_median, "\n")

# remove unused dataframe and free the memory
rm(
  test_imp_median,
  train_imp_median,
  val_imp_median,
  train_ind,
  val_ind,
  train_imp_result,
  val_imp_result,
  test_imp_result,
  test_pred_median,
  val_pred_median
)
gc()

# KNN imputation Methods
# Function for scaling data
scale_data <-
  function(x_train,
           x_val,
           x_test,
           method = c("center", "scale")) {
    preProc <- preProcess(x_train, method = method)
    train_data_scaled <- predict(preProc, x_train)
    val_data_scaled <- predict(preProc, x_val)
    test_data_scaled <- predict(preProc, x_test)
    
    return(list(
      train = train_data_scaled,
      val = val_data_scaled,
      test = test_data_scaled
    ))
  }

# Scale data
scaled_data <- scale_data(x_train, x_val, x_test)
train_data_scaled <- cbind(scaled_data$train, y_train)
val_data_scaled <- cbind(scaled_data$val, y_val)
test_data_scaled <- cbind(scaled_data$test, y_test)

# Define response and predictors
response <- 'y_train'
predictors <- c(
  'zip_code',
  'docket_number',
  'latitude',
  'longitude',
  'community_board',
  'council_district',
  'census_tract',
  'bin',
  'bbl',
  'administrative_district',
  'total_enrollment',
  'month_sin',
  'month_cos',
  'day_sin',
  'day_cos',
  'encoded_year',
  'name_count',
  'borough_bronx',
  'borough_brooklyn',
  'borough_manhattan',
  'borough_queens',
  'borough_staten_island',
  'residential_commercial_commercial',
  'residential_commercial_residential'
)

# Define a range of k values
k_num <- seq(1, 30, by = 2)
# initialize an empty vector that has same length of k_num to store the values of loop
val_errors <- numeric(length(k_num))

# Loop through k values, perform KNN imputation, train a model, and evaluate performance
for (i in seq_along(k_num)) {
  k <- k_num[i]
  
  # Perform KNN imputation
  train_data_knn_imp <- knnImputation(train_data_scaled, k = k)
  val_data_knn_imp <- knnImputation(val_data_scaled, k = k)
  
  # Train a ranger model
  ranger_model <- ranger(
    formula = as.formula(paste(
      response, "~", paste(predictors, collapse = "+")
    )),
    data = train_data_knn_imp,
    num.trees = 500,
    seed = 1234
  )
  
  # Calculate performance on the validation set
  val_predictions <- predict(ranger_model, val_data_knn_imp)
  val_predictions_vec <-
    val_predictions$predictions # Extract the numeric vector of predictions
  val_error <-
    RMSE(val_predictions_vec, val_data_knn_imp$y_val)
  val_errors[i] <- val_error
}

# Find the optimal k value
optimal_k <- k_num[which.min(val_errors)]

# to save run time for loop, save the variable to RDS object
saveRDS(optimal_k, 'optimal_k')

# Perform KNN imputation on the scaled datasets
train_imp_knn <- knnImputation(train_data_scaled, k = optimal_k)
test_imp_knn <- knnImputation(test_data_scaled, k = optimal_k)
val_imp_knn <- knnImputation(val_data_scaled, k = optimal_k)

# Train the ranger model on the imputed training set
model_knn <- ranger(y_train ~ ., data = train_imp_knn)

# Make predictions on the imputed test and validation sets
test_pred_knn <- predict(model_knn, test_imp_knn)$predictions
val_pred_knn <- predict(model_knn, val_imp_knn)$predictions

# Calculate and print RMSE for the test and validation sets
test_rmse_knn <- RMSE(test_pred_knn, y_test, "Test")
val_rmse_knn <- RMSE(val_pred_knn, y_val, "Validation")

# Print RMSE values
cat("Test RMSE for knn:", test_rmse_knn, "\n")
cat("Validation RMSE for knn:", val_rmse_knn, "\n")

# remove unused dataframe and free the memory
rm(
  model_knn,
  ranger_model,
  scaled_data,
  test_data_scaled,
  test_imp_knn,
  train_data_knn_imp,
  train_data_scaled,
  train_imp_knn,
  val_data_scaled,
  val_data_knn_imp,
  val_predictions,
  val_imp_knn
)
gc()

# missranger imputation
n_core <- detectCores()
n_core

best_rmse_combiner <- function(current_best, new_result) {
  if (is.null(current_best) || new_result$rmse < current_best$rmse) {
    return(new_result)
  } else {
    return(current_best)
  }
}

#
random_search_missranger <-
  function(data,
           target,
           n_iter = 10,
           n_cores = 15,
           seed = 1234) {
    set.seed(seed)
    doParallel::registerDoParallel(cores = n_cores)
    
    results <-
      foreach(i = 1:n_iter,
              .packages = c("missRanger", "caret")) %do% {
                # Randomly sample parameter values
                p_num_trees <- sample(50:1000, 1)
                p_min_node_size <- sample(1:10, 1)
                p_pmm_k <- sample(3:10, 1)
                
                # Impute missing values using missRanger with the random parameters
                imputed_data <-
                  missRanger(
                    data,
                    num.trees = p_num_trees,
                    min.node.size = p_min_node_size,
                    pmm.k = p_pmm_k,
                    num.threads = 1,
                    verbose = TRUE
                  )
                
                # Use K-fold cross-validation to compute the RMSE values
                resampling <-
                  trainControl(
                    method = "cv",
                    number = 5,
                    savePredictions = "final",
                    summaryFunction = defaultSummary
                  )
                
                model_cv <-
                  caret::train(
                    formula = as.formula(paste(target, "~.", sep = " ")),
                    data = imputed_data,
                    method = "ranger",
                    trControl = resampling
                  )
              
                rmse_values <- model_cv$resample$RMSE
                
                # Compute the average RMSE
                rmse <- mean(rmse_values)
                
                # Return the RMSE and the parameters for this iteration
                list(
                  rmse = rmse,
                  params = list(
                    num.trees = p_num_trees,
                    min.node.size = p_min_node_size,
                    pmm.k = p_pmm_k
                  )
                )
              }
    
    # Find the best parameters based on the lowest RMSE
    best_ind <- which.min(sapply(results, function(x)
      x$rmse))
    best_params <- results[[best_ind]]$params
    best_rmse <- results[[best_ind]]$rmse
    
    list(best_params = best_params, best_rmse = best_rmse)
  }

# Find the best parameters in the train set
# find the best parameters in train set
best_params <-
  random_search_missranger(
    data = train_enc,
    target = 'economic_need_index',
    n_iter = 10,
    n_cores = 15
  )

# save params locally in case need to rerun again
saveRDS(best_params, 'best_params')
print(best_params)

# Extract the best parameters and RMSE
best_params <- result$best_params
best_rmse <- result$best_rmse

# Print the best parameters and RMSE
cat("Best parameters:", "\n")
print(best_params)
cat("Best RMSE:", best_rmse, "\n")

# Impute missing values for train, test, and validation datasets
datasets <- list(train_enc, test_enc, val_enc)
names(datasets) <- c("train", "test", "val")

imputed_datasets <- lapply(datasets, function(dataset) {
  missRanger(
    dataset,
    num.trees = best_params$best_params$num.trees,
    min.node.size = best_params$best_params$min.node.size,
    pmm.k = best_params$best_params$pmm.k
  )
})

# Train a random forest model on the imputed_train dataset
model <-
  ranger(economic_need_index ~ ., data = imputed_datasets$train)

# Evaluate the model's performance on the imputed_test and imputed_validation datasets
rmse_results <- sapply(c("test", "val"), function(dataset_name) {
  predictions <-
    predict(model, data = imputed_datasets[[dataset_name]])$predictions
  rmse_val <-
    RMSE(predictions, imputed_datasets[[dataset_name]]$economic_need_index)
  cat(paste(
    "RMSE for",
    dataset_name,
    "using missRanger imputation:",
    rmse_val,
    "\n"
  ))
  return(rmse_val)
})

# Display RMSE summary for different imputation methods
cat(
  'Test RMSE using median imp:',
  rmse_test_median,
  'Validation RMSE using median imp:',
  rmse_val_median,
  '\n'
)
cat(
  'Test RMSE using knn imp:',
  test_rmse_knn,
  'Validation RMSE using knn imp:',
  val_rmse_knn,
  '\n'
)
cat(
  "Test RMSE for missRanger:",
  rmse_results["test"],
  "Validation RMSE for missRanger:",
  rmse_results["val"],
  "\n"
)

# end of this project
toc()
