
# Set Seed
set.seed(5275)

# Load necessary libraries
library(glmnet)
library(xgboost)

###########################################
# Function Definitions
###########################################

# Winsorization
winsorize = function(X) {
  winsor.vars = c("Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF", "Second_Flr_SF", 'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF", "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val")
  quan.value = 0.95
  for(var in winsor.vars){
    tmp = X[, var]
    myquan = quantile(tmp, probs = quan.value, na.rm = TRUE)
    tmp[tmp > myquan] =  myquan
    X[, var] = tmp
  }
  X
}

# Remove unnecessary variables
remove_vars = function(X) {
  remove.var = c('PID', 'Sale_Price', 'Street', 'Utilities', 'Condition_2', 'Roof_Matl', 'Heating', 'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude','Latitude')
  retain.var = !colnames(X) %in% remove.var
  X[, retain.var]
}

# Categorical Vars to dummy variables
convert_categorical_vars = function(X) {
  
  categorical.vars = colnames(X)[which(sapply(X, function(x) mode(x)=="character"))]
  
  X.matrix <- X[, !colnames(X) %in% categorical.vars, drop=FALSE]
  
  n <- nrow(X.matrix)
  for(var in categorical.vars){
    mylevels <- sort(unique(X[, var]))
    m <- length(mylevels)
    m <- ifelse(m>2, m, 1)
    tmp <- matrix(0, n, m)
    col.names <- NULL
    for(j in 1:m){
      tmp[X[, var]==mylevels[j], j] <- 1
      col.names <- c(col.names, paste(var, '_', mylevels[j], sep=''))
    }
    colnames(tmp) <- col.names
    X.matrix <- cbind(X.matrix, tmp)
  }
  
  X.matrix
  
}

# Synchronize vars from train and test
sync_vars = function(train, test) {
  col.names.train = colnames(train)
  col.names.test =  colnames(test)
  cols.add = col.names.train[!col.names.train %in% col.names.test]
  for(col.name in cols.add) {
    test[,col.name] = 0
  }
  test[,col.names.train]
}

# Replace NAs with zeros
replace_missing = function(X) {
  for(i in 1:ncol(X)) {
    X[is.na(X[,i]),i] = 0
  }
  X
}

# Predict using Lasso/Ridge
lasso_predictions = function (train.matrix, train.y, test.matrix) {
  set.seed(5275)
  train.x = as.matrix(train.matrix)
  test.x = as.matrix(test.matrix)
  cv.out = cv.glmnet(train.x, train.y, alpha = 1)
  sel.vars = predict(cv.out, type="nonzero", s = cv.out$lambda.min)[,1]
  train.x = as.matrix(train.matrix[, sel.vars])
  cv.out = cv.glmnet(train.x, train.y, alpha = 0)
  test.x = as.matrix(test.matrix[, sel.vars])
  predict(cv.out, s = cv.out$lambda.min, newx = test.x)
  
}

# Predict using XGBoost
boosting_predictions = function(train.matrix, train.y, test.matrix) {
  
  train.x = as.matrix(train.matrix)
  test.x = as.matrix(test.matrix)
  
  xgb.model = xgboost(data = train.x, 
                      label = train.y, 
                      max_depth = 3,
                      eta = 0.05, 
                      nrounds = 5000,
                      min_child_weight = 4,
                      verbose = FALSE)
  
  predict(xgb.model, newdata = test.x)
}

###########################################
# Step 1: Preprocess training data
###########################################

# load train.csv
train = read.csv("train.csv", stringsAsFactors = FALSE)

# pre-process training data
train.x = remove_vars(train)
train.x = replace_missing(train.x)
train.x = winsorize(train.x)
train.x = convert_categorical_vars(train.x)

train.y = log(train[,"Sale_Price"])

###########################################
# Step 2: Preprocess test data
###########################################

# load test.csv
test = read.csv("test.csv", stringsAsFactors = FALSE)

# pre-process test data
test.x = remove_vars(test)
test.x = replace_missing(test.x)
test.x = winsorize(test.x)
test.x = convert_categorical_vars(test.x)

# ensure that train and test have the same set of columns
test.x = sync_vars(train.x, test.x)

###########################################
# Step 3: Make predictions 
#         and output predictions into two files
###########################################

# Make predictions using Linear Regression Lasso/Ridge
lasso.pred = lasso_predictions(train.x, train.y, test.x)

# Make predictions using XGBoost
boost.pred = boosting_predictions(train.x, train.y, test.x)

mod1.pred = cbind(test[,1], exp(lasso.pred[,1]))
mod2.pred = cbind(test[,1], exp(boost.pred))

colnames(mod1.pred) = c("PID","Sale_Price")
colnames(mod2.pred) = c("PID","Sale_Price")

# Save submissions
write.csv(mod1.pred,"mysubmission1.txt", row.names = FALSE)
write.csv(mod2.pred,"mysubmission2.txt", row.names = FALSE)

# Test submissions
# test.y = read.csv("test_y.csv")
# pred = read.csv("mysubmission1.txt")
# names(test.y)[2] = "True_Sale_Price"
# pred = merge(pred, test.y, by="PID")
# sqrt(mean((log(pred$Sale_Price) - log(pred$True_Sale_Price))^2))

# pred = read.csv("mysubmission2.txt")
# names(test.y)[2] = "True_Sale_Price"
# pred = merge(pred, test.y, by="PID")
# sqrt(mean((log(pred$Sale_Price) - log(pred$True_Sale_Price))^2))
