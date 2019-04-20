import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import boxcox1p
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor
import warnings

# Ignore warnings from sklearn
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

filepath = "/Users/alekx/Desktop/Kaggle/house-prices-advanced-regression-techniques"
train = pd.read_csv(filepath + "/train.csv")
test = pd.read_csv(filepath + "/test.csv")
test_ID = test['Id']
# Checking for outlier data, as author of HPP dataset suggested
fix, ax = plt.subplots()
ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
# Delete outliers as shown in plot, >4000 GrLivArea and < 300000 SalePrice
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

# Find correlation on features that are a numerical value
train_corr = train.select_dtypes(include=[np.number])
del train_corr['Id']
corr = train_corr.corr()
plt.subplots(figsize=(20, 9))
sns.heatmap(corr, annot=True)

# Select those that have correlation with SalePrice > 50%
top_feature = corr.index[abs(corr['SalePrice'] > 0.5)]
plt.subplots(figsize=(8, 8))
top_corr = train[top_feature].corr()
sns.heatmap(top_corr, annot=True)

col = [
        'SalePrice', 'OverallQual', 'GrLivArea',
        'GarageCars', 'TotalBsmtSF', 'FullBath',
        'TotRmsAbvGrd', 'YearBuilt'
]
sns.set(style='ticks')
sns.pairplot(train[col], size=3, kind='reg')

sns.scatterplot(train['GrLivArea'], train['TotalBsmtSF'])

# Define function to check for skewness of feature
def check_skew(col):
    plt.subplots(figsize=(12, 9))
    sns.distplot(train[col], fit=stats.norm)
    # Get the fitted parameters used by the function
    (mu, sigma) = stats.norm.fit(train[col])
    # Plot with the distribution
    plt.legend(['Normal Distr. ($mu=$ {:.2f} and $sigma=$ {:.2f})'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')

# Check target output SalePrice skewness
check_skew('SalePrice')

# Probability plot
stats.probplot(train['SalePrice'], plot=plt)
plt.show()

# Log transform for output target, y
# Note: All other features that are skewed needs to be transformed as well
train['SalePrice'] = np.log1p(train['SalePrice'])

# Will be used later on to extract and separate the training and test data
train_num = train.shape[0]
test_num = test.shape[0]

y = train.SalePrice.values

# Dropping PoolQC, MiscFeature, MiscVal, Alley, Fence, FireplaceQu due to many missing values
drop_columns = ['Id', 'PoolQC', 'Alley', 'Fence', 'RoofStyle', 'RoofMatl', 'Exterior2nd', 'FireplaceQu', 'Street', 'LandContour']
train_data = train.drop(drop_columns + ['SalePrice'], axis=1)
test_data = test.copy().drop(drop_columns, axis=1)
frames = [train_data, test_data]
combined = pd.concat(frames).reset_index(drop=True)

# Check feature counts, especially those with a lot of zeros
# combined['Functional'].value_counts()

# Check missing values
train.columns[train.isnull().any()]

# combined.PoolQC[combined.PoolQC.notnull()]

plt.figure(figsize=(15, 8))
sns.heatmap(train.isnull())
plt.show()

Isnull = train.isnull().sum() / len(train) * 100
Isnull = Isnull[Isnull > 0]
Isnull.sort_values(inplace=True, ascending=False)
Isnull = Isnull.to_frame()
Isnull.columns = ['Percent of missing value, %']
Isnull.index.names = ['Features']
Isnull['Features'] = Isnull.index
plt.figure(figsize=(13, 8))
sns.set(style='darkgrid')
sns.barplot(x='Features', y='Percent of missing value, %', data=Isnull)
plt.title('Percent missing data by feature', fontsize=15)
plt.xticks(rotation=60)
plt.show()

# Handle categorical missing values by filling NAN or null numbers as "None"
# combined['PoolQC'] = combined['PoolQC'].fillna("None")
# combined['MiscFeature'] = combined['MiscFeature'].fillna('None')
# combined['Alley'] = combined['Alley'].fillna('None')
# combined['Fence'] = combined['Fence'].fillna('None')
# combined['FireplaceQu'] = combined['FireplaceQu'].fillna('None')

# Assume LotFrontage to be similar with surrounding neighborhoods, group by neighborhood and taken median value
combined['LotFrontage'] = combined.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# If no garage, all other garage columns are none
garage_col = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in garage_col:
    combined[col] = combined[col].fillna('None')

garage_col_all = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'GarageArea', 'GarageCars']
combined.groupby('GarageType')[garage_col_all].count()
# If no basement, all other basement columns are none, categorical data = None
# If no basement, all other basement columns are 0, integer data = 0
bsmt_col_int = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
for col in bsmt_col_int:
    combined[col] = combined[col].fillna(0)
bsmt_col_cat = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
for col in bsmt_col_cat:
    combined[col] = combined[col].fillna('None')

combined['MasVnrType'] = combined['MasVnrType'].fillna('None')
combined['MasVnrArea'] = combined['MasVnrArea'].fillna(0)

# For MSZoning, the cell needs to be filled and will be assumed the most common which is RL
combined['MSZoning'].value_counts()
combined['MSZoning'] = combined['MSZoning'].fillna(combined['MSZoning'].mode()[0])

# Since most utilities fall under AllPub category with a single exception, drop feature
combined['Utilities'].value_counts()
combined = combined.drop(['Utilities'], axis=1)

# Removed Fence, Electrical, Exterior2nd
remainder_col = ['KitchenQual', 'Exterior1st', 'SaleType']
for col in remainder_col:
    combined[col] = combined[col].fillna(combined[col].mode()[0])

# Assume if no value then it is typical
combined['Functional'] = combined['Functional'].fillna('Typ')

# Check if there is any remainder null columns
combined.columns[combined.isnull().any()]
# Change some features that have numerical values but are actually categorical data, convert to string
# We do not want the model to interpret numerical relationship such as 1<5<10
# MSSubclass, YrSold, MoSold, OverallCond are some examples
combined['MSSubClass'] = combined['MSSubClass'].apply(str)
combined['OverallCond'] = combined['OverallCond'].apply(str)
combined['YrSold'] = combined['YrSold'].apply(str)
combined['MoSold'] = combined['MoSold'].apply(str)

# Encoding for categorical data
# Removed FireplaceQu, PoolQC, Fence, Alley, Street, CentralAir
obj_cols = (
                    'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'ExterQual',
                    'ExterCond', 'HeatingQC', 'KitchenQual', 'BsmtFinType1', 'BsmtFinType2',
                    'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope', 'LotShape', 'PavedDrive',
                    'CentralAir', 'MSSubClass', 'OverallCond', 'YrSold', 'MoSold', 'YearBuilt', 'YearRemodAdd'
)

for feature in obj_cols:
    lblenc = LabelEncoder()
    lblenc.fit(list(combined[feature].values))
    combined[feature] = lblenc.transform(list(combined[feature].values))

# Add additional feature of total sqft
combined['TotalSF'] = combined['TotalBsmtSF'] + combined['1stFlrSF'] + combined['2ndFlrSF']

# Adjust for features that show high skewness, 0.75
numeric_feats = combined.dtypes[combined.dtypes != "object"].index
skewed_feats = combined[numeric_feats].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)
print('\nSkew in numerical features: \n')
skewness = pd.DataFrame({'Skew': skewed_feats})
skewness = skewness[abs(skewness) > 0.5]
skewed_features = skewness.index
skewed_features
lam = 0.15
for feat in skewed_features:
    combined[feat] = boxcox1p(combined[feat], lam)

# run one hot encoding for remainder non numerical data, dataset can't be object or string dtypes
dummy_combined = pd.get_dummies(combined)
dummy_combined.shape

# Extract from the combined data into train and test
train = dummy_combined[:train_num]
test = dummy_combined[train_num:]

# Function to run cross validation
def cross_val(model):
    n_folds = 5
    kf = KFold(n_folds, shuffle=True, random_state=0).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y, scoring='neg_mean_squared_error', cv=kf))
    return rmse

# Train test split for XGBoost
train_X, test_X, train_y, test_y = train_test_split(train, y, test_size=0.2, random_state=0)

# XGBoost Model
xgb_model = XGBRegressor(
                    learn_rate=0.01,
                    n_estimators=500,
                    max_depth=3,
                    min_child_weight=1,
                    subsample=0.6,
                    colsample_bytree=1,
                    gamma=0,
                    reg_alpha=0.11,
                    random_state=0
)

# Fit to XGBoost model
xgb_model.fit(train_X, train_y, early_stopping_rounds=10, eval_set=[(test_X, test_y)], verbose=True)
xgb_predict = xgb_model.predict(test)

# Cross Validation
scores = cross_val(xgb_model)
print("Cross Value MAE %.6f" % (scores.mean()))

# Reverse log to SalePrice predictions
reverse_log_xgb = np.expm1(xgb_predict)

# Create a new dataframe for submission into Kaggle HPP Competition
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = reverse_log_xgb
sub.to_csv('submission.csv', index=False)
