# Kaggle Competition - Housing Price Prediction
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

# Call in csv with desktop file path and put in a DataFrame
ames_file_path = "/Users/alekx/Desktop/Kaggle/house-prices-advanced-regression-techniques/train.csv"
ames_data = pd.read_csv(ames_file_path)

# Call in csv that would be later used to submit to Kaggle Housing Price Prediction Competition
test_file_path = "/Users/alekx/Desktop/Kaggle/house-prices-advanced-regression-techniques/test.csv"
test_data = pd.read_csv(test_file_path)

# Output prediction target, y
y = ames_data.SalePrice

# Features, X to build on for prediction (remove output target and float type features)
columns_to_drop = ['SalePrice', 'Id', 'LotFrontage', 'GarageYrBlt', 'MasVnrArea']
ames_copy_data = ames_data.copy().drop(columns_to_drop, axis=1)

test_columns_to_drop = ['Id', 'LotFrontage', 'GarageYrBlt', 'MasVnrArea']
test_copy_data = test_data.copy().drop(test_columns_to_drop, axis=1)

# Concatenate dataframes for preprocessing and maintaining similar columns
frames = [ames_copy_data, test_copy_data]
combined_data = pd.concat(frames, keys=['1', '2'])

# Deciding the initial threshold to be 0.5% of dataset size
tot_instances = combined_data.shape[0]
threshold = tot_instances * 0.005

# Apply the count threshold to all the categorical values
obj_columns = list(combined_data.select_dtypes(include=['object']).columns)
combined_data = combined_data.apply(
                                lambda x: x.mask(x.map(x.value_counts()) < threshold, 'RARE')
                                if x.name in obj_columns else x
)

# Using Pandas for One Hot Encoding
dummy_data_ohe = pd.get_dummies(combined_data)

# Using the keys from the concatenated data, select training dataset and test dataset
train_X = dummy_data_ohe.loc['1']
test_X = dummy_data_ohe.loc['2']

# XGBoost model + fit
xgb_model = XGBRegressor(learn_rate=0.045, n_estimators=500, subsample=0.8, random_state=0)
xgb_model.fit(train_X, y, early_stopping_rounds=5, eval_set=[(train_X, y)], verbose=False)

# Cross-Validation to determine model quality
scores = cross_val_score(xgb_model, train_X, y, scoring='neg_mean_absolute_error')
print("With Categoricals Cross Value MAE %.2f" % (-1 * scores.mean()))

# Run prediction with test data and create submissions.csv file
xgb_predict = xgb_model.predict(test_X)
# np.savetxt("submission.csv", xgb_predict, header='SalePrice', delimiter=",")
