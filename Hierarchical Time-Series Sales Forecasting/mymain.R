library(lubridate)
library(xgboost)
library(tidyverse)
library(caret)
library(tidyr)

mypredict = function() {

  if (t == 5) {
    
    return(lm_predict(train))
    
  } else {
    predictions = bind_rows(xgboost_predict(train), lm_predict(train)) 
    
    # Average over two methods
    predictions = aggregate(Weekly_Pred ~ Date + Store + Dept, data = predictions, FUN = mean)
    
    return(predictions)
  }
}

shift_weeks = function(data) {
  
  data[data$Year == 2010, "Week"] = data[data$Year == 2010, "Week"] - 1
  
  return(data)
}

xgboost_predict = function(train) {
  
  startDate = as.Date("2011-01-01")
  endDate = as.Date("2011-03-01")
  
  startDate = ymd(startDate) %m+% months(t*2)
  endDate = ymd(endDate) %m+% months(t*2)
  
  trn = data.frame(train)
  trn$Year = lubridate::year(trn$Date)
  trn$Week = lubridate::week(trn$Date)
  trn$Month = lubridate::month(trn$Date)
  trn$Weights = ifelse(trn$IsHoliday == TRUE, 5, 1)
  trn = shift_weeks(trn)
  
  tst = test[test$Date >= startDate & test$Date < endDate, ]
  tst$Year = lubridate::year(tst$Date)
  tst$Week = lubridate::week(tst$Date)
  tst$Month = lubridate::month(tst$Date)
  tst = shift_weeks(tst)
  
  dtrain = xgb.DMatrix(as.matrix(trn[, c("Store","Dept", "IsHoliday", "Year","Week","Month")]), label = trn$Weekly_Sales, weight = trn$Weights)
  
  test.x = as.matrix(tst[, c("Store","Dept", "IsHoliday", "Year","Week","Month")])
  
  fit = xgb.train(data = dtrain,
                  booster = "gbtree",
                  nrounds = 200,
                  verbose = 0,
                  max_depth = 20,
                  eta = 0.02, 
                  min_child_weight = 4,
                  eval_metric = "mae")
  
  tst$Weekly_Pred = predict(fit, newdata = test.x)
  
  return(tst[,  c('Date', 'Store', 'Dept','Weekly_Pred')])
  
}

preprocess.svd <- function(train, n.comp){
  
  # Source reference:
  # Adapted from https://github.com/davidthaler/Walmart_competition_code
  
  depts = unique(train$Dept) 
  
  new_data = NULL
  
  for (dept in depts) {
    
    dept_rows = train[train$Dept == dept, c("Store","Dept","Date","Weekly_Sales")]
    
    # Arrange data from a particular department as a matrix Xmxn where m denotes the number of 
    # stores that have this particular department and n is the number of weeks
    # dept_rows = dcast(dept_rows, formula = "Store + Dept ~ Date", value.var = "Weekly_Sales", drop = TRUE) 
    dept_rows = spread(dept_rows, Date, Weekly_Sales)
    
    if (nrow(dept_rows) > n.comp){
      
      dept_rows[is.na(dept_rows)] = 0
      
      # Subtract store means
      store_means = rowMeans(dept_rows[, 3:ncol(dept_rows)])
      dept_rows[, 3:ncol(dept_rows)] = dept_rows[, 3:ncol(dept_rows)] - store_means
      
      # Apply SVD for each set of dept rows, choosing the top 8 components
      z = svd(dept_rows[, 3:ncol(dept_rows)], nu=n.comp, nv=n.comp)
      s = diag(z$d[1:n.comp])
      dept_rows[, 3:ncol(dept_rows)] = data.frame(z$u %*% s %*% t(z$v))
      
      # Add store means
      dept_rows[, 3:ncol(dept_rows)] = dept_rows[, 3:ncol(dept_rows)] + store_means
      
      # Convert weekly sales columns back into rows
      # dept_rows = melt(dept_rows, id = c("Store","Dept")) 
      # colnames(dept_rows) = c("Store","Dept","Date","Weekly_Sales")
      dept_rows = gather(dept_rows, Date, Weekly_Sales, -Store, -Dept)
      
      # append these rows to our new data frame
      new_data = rbind(new_data, dept_rows)
    }
  }
  
  return(new_data)
}

lm_predict = function(train) {
  
  train = preprocess.svd(train, 8)
  
  start_date <- ymd("2011-03-01") %m+% months(2 * (t - 1))
  end_date <- ymd("2011-05-01") %m+% months(2 * (t - 1))
  test_current <- test %>%
    filter(Date >= start_date & Date < end_date) %>%
    select(-IsHoliday)
  
  start_last_year = min(test_current$Date) - 375
  end_last_year = max(test_current$Date) - 350
  
  tmp_train <- train %>%
    filter(Date > start_last_year & Date < end_last_year) %>%
    mutate(Wk = ifelse(year(Date) == 2010, week(Date)-1, week(Date))) %>%
    rename(Weekly_Pred = Weekly_Sales) %>%
    select(-Date)
  
  test_current <- test_current %>%
    mutate(Wk = week(Date))
  
  # find the unique pairs of (Store, Dept) combo that appeared in both training and test sets
  train_pairs <- train[, 1:2] %>% count(Store, Dept) %>% filter(n != 0)
  test_pairs <- test_current[, 1:2] %>% count(Store, Dept) %>% filter(n != 0)
  unique_pairs <- intersect(train_pairs[, 1:2], test_pairs[, 1:2])
  
  # pick out the needed training samples, convert to dummy coding, then put them into a list
  train_split <- unique_pairs %>% 
    left_join(train, by = c('Store', 'Dept')) %>% 
    mutate(Wk = factor(ifelse(year(Date) == 2010, week(Date) - 1, week(Date)), levels = 1:52)) %>% 
    mutate(Yr = year(Date))
  train_split = as_tibble(model.matrix(~ Weekly_Sales + Store + Dept + Yr + I(Yr^2) + Wk, train_split)) %>% group_split(Store, Dept)
  
  # do the same for the test set
  test_split <- unique_pairs %>% 
    left_join(test_current, by = c('Store', 'Dept')) %>% 
    mutate(Wk = factor(ifelse(year(Date) == 2010, week(Date) - 1, week(Date)), levels = 1:52)) %>% 
    mutate(Yr = year(Date))
  test_split = as_tibble(model.matrix(~ Store + Dept + Yr + I(Yr^2) + Wk, test_split)) %>% mutate(Date = test_split$Date) %>% group_split(Store, Dept)
  
  # pre-allocate a list to store the predictions
  test_pred <- vector(mode = "list", length = nrow(unique_pairs))
  
  # perform regression for each split, note we used lm.fit instead of lm
  for (i in 1:nrow(unique_pairs)) {
  
    tmp_train <- train_split[[i]]
    tmp_test <- test_split[[i]]
    
    # shift for fold 5
    if (t==5) {
      shift = 1/7
      for (wk in 48:51) {
        cur_week = paste("Wk",wk,sep="")
        next_week = paste("Wk",wk+1,sep="")
        tmp_test[, cur_week] = tmp_test[, cur_week] * (1 - shift) + tmp_test[, next_week] * shift
      }
      tmp_test[, "Wk52"] = tmp_test[, "Wk52"] * (1 - shift) 
    }   
    
    mycoef <- lm.fit(as.matrix(tmp_train[, -(2:4)]), tmp_train$Weekly_Sales)$coefficients
    mycoef[is.na(mycoef)] <- 0
    tmp_pred <- mycoef[1] + as.matrix(tmp_test[, 4:56]) %*% mycoef[-1]
    
    test_pred[[i]] <- cbind(tmp_test[, 2:3], Date = tmp_test$Date, Weekly_Pred = tmp_pred[,1])
  }
  
  # turn the list into a table at once, 
  # this is much more efficient then keep concatenating small tables
  test_pred <- bind_rows(test_pred)
  
  return(test_pred)
  
}
