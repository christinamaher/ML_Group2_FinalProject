install.packages("e1071", repos = "https://cloud.r-project.org/")
install.packages("caret", repos = "https://cloud.r-project.org/")
install.packages("pROC", repos = "https://cloud.r-project.org/")
install.packages("e1071", dependencies = TRUE)
install.packages("caret", dependencies = TRUE)
install.packages("pROC", dependencies = TRUE)
library(e1071)
library(caret)
library(pROC)

# Load necessary libraries
library(e1071)      # For SVM model
library(caret)      # For model evaluation
library(ggplot2)    # For ROC Curve
library(dplyr)      # Data manipulation
library(pROC)       # ROC and AUC calculation

# Load the dataset
file_path <- "simulated_patient_data.csv"  # Update the path if needed
simulated_data <- read.csv(file_path)

# Data Preprocessing

# 1. Convert categorical variables to numeric
simulated_data$gender <- ifelse(simulated_data$gender == "Male", 1, 0)
simulated_data$race <- as.numeric(factor(simulated_data$race))

# 2. Aggregate time-series data (mean, std, slope)
aggregate_features <- function(df, var_name) {
  mean_col <- rowMeans(df[, grep(var_name, colnames(df))])
  std_col <- apply(df[, grep(var_name, colnames(df))], 1, sd)
  slope_col <- apply(df[, grep(var_name, colnames(df))], 1, function(x) coef(lm(x ~ seq_along(x)))[2])
  data.frame(mean = mean_col, std = std_col, slope = slope_col)
}

# Aggregating heart rate and hematocrit data
heart_rate_features <- aggregate_features(simulated_data, "heart_rate")
hematocrit_features <- aggregate_features(simulated_data, "hematocrit")

# Combine static and aggregated features
features <- cbind(simulated_data[, c("age", "gender", "race")], heart_rate_features, hematocrit_features)
target <- simulated_data$readmitted

# Convert the target variable to factor for classification
target <- as.factor(target)

# Train-Test Split (80-20)
set.seed(42)
trainIndex <- createDataPartition(target, p = 0.8, list = FALSE)
trainData <- features[trainIndex, ]
testData <- features[-trainIndex, ]
trainLabels <- target[trainIndex]
testLabels <- target[-trainIndex]

# Model Training: SVM with RBF kernel
svm_model <- svm(trainLabels ~ ., data = trainData, kernel = "radial", probability = TRUE)

# Model Prediction
pred <- predict(svm_model, testData, probability = TRUE)
prob <- attr(pred, "probabilities")[, 2]

# Model Evaluation
accuracy <- mean(pred == testLabels)
conf_matrix <- table(Predicted = pred, Actual = testLabels)

# Calculate F1 Score
f1 <- F1_Score(pred, testLabels)

# ROC and AUC
roc_obj <- roc(as.numeric(testLabels), prob)
auc_value <- auc(roc_obj)

# Print Evaluation Metrics
cat("Accuracy:", round(accuracy * 100, 2), "%\n")
cat("F1 Score:", round(f1, 2), "\n")
cat("AUC:", round(auc_value, 2), "\n")

# Plotting ROC Curve
ggroc(roc_obj, color = "blue") +
  ggtitle("SVM ROC Curve for ICU Readmission Prediction") +
  xlab("False Positive Rate") +
  ylab("True Positive Rate") +
  theme_minimal()
