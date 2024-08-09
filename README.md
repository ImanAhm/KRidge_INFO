# KRidge_INFO
This code implements a Kernel Ridge Regression (KRidge) model optimized using the **INFO** (Weighted Mean of Vectors Optimization) algorithm. The provided implementation focuses on predicting a target variable from a dataset and evaluating its performance using various metrics. 

## Features  

- **KRidge Regression:** A powerful non-linear regression technique using kernel functions to capture complex relationships in data.  
- **INFO Optimization:** Employs the INFO algorithm to determine optimal hyperparameters for the KRidge model, enhancing its predictive accuracy.  
- **Performance Evaluation:** Calculates and saves a comprehensive set of regression metrics (R2, RMSE, MAPE, KGE, NSE, WHD, VSD, WAI) to assess model performance on training and testing datasets.  
- **Excel Output:** Saves the calculated metrics and actual vs. predicted values into a well-structured Excel file for easy analysis and comparison.  

************************************
# Installation

- **`pip install KRidge_INFO`**
- **`pip install IM_Metrics`**
- **`pip install info_optimizer`**

************************************



**`Eample`:** 

import pandas as pd  
from sklearn.model_selection import train_test_split  
from IM_Metrics import Save_Metrics
from KRidge_INFO.KRidge_INFO import RUN_INFO,PredictedValue_TrainTest 

# Read data  
data = pd.read_excel('Data.xlsx')   
nTs = 0.3  # Percentage of test dataset     
X = data.iloc[:, :-1].values  
y = data.iloc[:, -1].values  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=nTs, random_state=0)   
 
kernel_type = 'wavelet'  

# Main parameters of INFO and KRidge
nP = 50, # population Size of INFO Algorithm
MaxIt = 20 # Maximum iteration of INFO Algorithm

UC = 2e10    #Upper Bound for C coefficient in KRidge model.

UKF = 2e10   #Upper Bound for kernel function coefficient in KRidge model.

# RUN Main Model
best_parameters = RUN_INFO(nP, MaxIt,X_train, X_test, y_train, y_test,kernel_type,UC,UKF)

# After obtaining final predictions  

y_train_pred, y_train,y_test_pred,y_test = PredictedValue_TrainTest(best_parameters, kernel_type, 
                            X_train, y_train,X_test, y_test)
# Save Results
metrics_filename = 'Results of KRidge.xlsx'

Save_Metrics(y_train, y_train_pred, y_test, y_test_pred,metrics_filename)

