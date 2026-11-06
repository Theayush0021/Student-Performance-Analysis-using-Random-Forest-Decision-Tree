# Student Performance Analysis using Random Forest & Decision Tree


# 1. Load Packages

library(randomForest)
library(rpart)


# 2. Create Output Folders

dir.create(
  "Day11_Outputs",
  showWarnings = FALSE
)

dir.create(
  "Day11_Outputs/confusion_matrices",
  showWarnings = FALSE,
  recursive = TRUE
)


# 3. Load Dataset

students <- read.csv("StudentsPerformance.csv")


# 4. Data Preparation

students$average_score <- (
  students$math.score +
    students$reading.score +
    students$writing.score
) / 3

students$result <- factor(
  ifelse(
    students$average_score >= 60,
    "Pass",
    "Needs Improvement"
  ),
  levels = c("Needs Improvement", "Pass")
)

# Select Features

data <- students[, c(
  "gender",
  "race.ethnicity",
  "parental.level.of.education",
  "lunch",
  "test.preparation.course",
  "result"
)]


# 5. Train-Test Split (80:20)

set.seed(123)

train_index <- unlist(
  lapply(
    levels(data$result),
    function(cl) {
      
      class_rows <- which(data$result == cl)
      
      sample(
        class_rows,
        size = floor(0.8 * length(class_rows))
      )
    }
  )
)

train_data <- data[train_index, ]
test_data <- data[-train_index, ]

cat("Training Rows:", nrow(train_data))
cat("\nTesting Rows:", nrow(test_data))


# 6. Random Forest Model

set.seed(123)

rf_model <- randomForest(
  result ~ .,
  data = train_data,
  ntree = 500,
  importance = TRUE
)

# Random Forest Prediction

rf_pred <- predict(
  rf_model,
  test_data
)

# Random Forest Accuracy

rf_accuracy <- mean(
  rf_pred == test_data$result
) * 100


# 7. Feature Importance

feature_importance <- importance(rf_model)

print(feature_importance)

# Convert into Data Frame

feature_results <- data.frame(
  Feature = rownames(feature_importance),
  feature_importance,
  row.names = NULL
)

# Save Feature Importance

write.csv(
  feature_results,
  "Day11_Outputs/feature_importance.csv",
  row.names = FALSE
)

# Feature Importance Graph

png(
  "Day11_Outputs/feature_importance.png",
  width = 1400,
  height = 900
)

varImpPlot(
  rf_model,
  main = "Random Forest Feature Importance"
)

dev.off()


# 8. Original Decision Tree

tree_model <- rpart(
  result ~ .,
  data = train_data,
  method = "class",
  control = rpart.control(cp = 0.001)
)

# Display CP Table

printcp(tree_model)

# Save CP Table

cp_results <- as.data.frame(
  tree_model$cptable
)

write.csv(
  cp_results,
  "Day11_Outputs/decision_tree_cp_table.csv",
  row.names = FALSE
)


# 9. Find Best CP and Prune Tree

best_cp <- tree_model$cptable[
  which.min(tree_model$cptable[, "xerror"]),
  "CP"
]

cat("\nBest CP:", best_cp)

pruned_model <- prune(
  tree_model,
  cp = best_cp
)


# 10. Compare Tree Size

original_size <- nrow(tree_model$frame)

pruned_size <- nrow(pruned_model$frame)

cat("\nOriginal Tree Size:", original_size)

cat("\nPruned Tree Size:", pruned_size)


# 11. Save Original Tree Graph

png(
  "Day11_Outputs/original_decision_tree.png",
  width = 1400,
  height = 1000
)

plot(tree_model)
text(tree_model, use.n = TRUE)

dev.off()


# 12. Save Pruned Tree Graph

png(
  "Day11_Outputs/pruned_decision_tree.png",
  width = 1400,
  height = 1000
)

plot(pruned_model)
text(pruned_model, use.n = TRUE)

dev.off()


# 13. Model Predictions

# Original Tree Prediction

original_pred <- predict(
  tree_model,
  test_data,
  type = "class"
)

# Pruned Tree Prediction

pruned_pred <- predict(
  pruned_model,
  test_data,
  type = "class"
)


# 14. Training Accuracy

# Original Tree

original_train_pred <- predict(
  tree_model,
  train_data,
  type = "class"
)

original_train_accuracy <- mean(
  original_train_pred == train_data$result
) * 100

# Pruned Tree

pruned_train_pred <- predict(
  pruned_model,
  train_data,
  type = "class"
)

pruned_train_accuracy <- mean(
  pruned_train_pred == train_data$result
) * 100

# Random Forest Training Accuracy

rf_train_pred <- predict(
  rf_model,
  train_data
)

rf_train_accuracy <- mean(
  rf_train_pred == train_data$result
) * 100


# 15. Test Accuracy

# Original Tree

original_test_accuracy <- mean(
  original_pred == test_data$result
) * 100

# Pruned Tree

pruned_test_accuracy <- mean(
  pruned_pred == test_data$result
) * 100


# 16. Model Comparison

results <- data.frame(
  
  Model = c(
    "Random Forest",
    "Original Decision Tree",
    "Pruned Decision Tree"
  ),
  
  Train_Accuracy = c(
    rf_train_accuracy,
    original_train_accuracy,
    pruned_train_accuracy
  ),
  
  Test_Accuracy = c(
    rf_accuracy,
    original_test_accuracy,
    pruned_test_accuracy
  ),
  
  Tree_Size = c(
    NA,
    original_size,
    pruned_size
  )
)

print(results)

# Save Results

write.csv(
  results,
  "Day11_Outputs/model_comparison_results.csv",
  row.names = FALSE
)


# 17. Confusion Matrices

rf_cm <- table(
  Actual = test_data$result,
  Predicted = factor(
    rf_pred,
    levels = levels(data$result)
  )
)

original_cm <- table(
  Actual = test_data$result,
  Predicted = factor(
    original_pred,
    levels = levels(data$result)
  )
)

pruned_cm <- table(
  Actual = test_data$result,
  Predicted = factor(
    pruned_pred,
    levels = levels(data$result)
  )
)

# Display Confusion Matrices

print(rf_cm)
print(original_cm)
print(pruned_cm)

# Save Confusion Matrices

write.csv(
  as.data.frame.matrix(rf_cm),
  "Day11_Outputs/confusion_matrices/random_forest.csv"
)

write.csv(
  as.data.frame.matrix(original_cm),
  "Day11_Outputs/confusion_matrices/original_tree.csv"
)

write.csv(
  as.data.frame.matrix(pruned_cm),
  "Day11_Outputs/confusion_matrices/pruned_tree.csv"
)


# 18. Save Model Predictions

prediction_results <- data.frame(
  Actual = test_data$result,
  Random_Forest = rf_pred,
  Original_Tree = original_pred,
  Pruned_Tree = pruned_pred
)

write.csv(
  prediction_results,
  "Day11_Outputs/model_predictions.csv",
  row.names = FALSE
)


# 19. Accuracy Comparison Graph

png(
  "Day11_Outputs/model_accuracy_comparison.png",
  width = 1200,
  height = 800
)

barplot(
  results$Test_Accuracy,
  names.arg = results$Model,
  main = "Model Test Accuracy Comparison",
  xlab = "Machine Learning Models",
  ylab = "Accuracy (%)",
  ylim = c(0, 100),
  las = 2
)

dev.off()


# 20. Best Model Report

best_model <- results[
  which.max(results$Test_Accuracy),
]

print(best_model)

write.csv(
  best_model,
  "Day11_Outputs/best_model_report.csv",
  row.names = FALSE
)


# 21. Final Results
cat("\n----------------------------------")


cat("\nRandom Forest Test Accuracy:",
    rf_accuracy, "%")

cat("\nOriginal Tree Test Accuracy:",
    original_test_accuracy, "%")

cat("\nPruned Tree Test Accuracy:",
    pruned_test_accuracy, "%")

cat("\nOriginal Tree Size:",
    original_size)

cat("\nPruned Tree Size:",
    pruned_size)

cat("\n\nBest Model:",
    as.character(best_model$Model))

cat("\n\nAll CSV files and graphs saved successfully!")

