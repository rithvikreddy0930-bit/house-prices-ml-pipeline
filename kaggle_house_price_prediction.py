# IMPORTS

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
import warnings

# SETTINGS

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

# LOAD DATA

df = pd.read_csv('train.csv')

# MISSING VALUE HANDLING

df['LotFrontage'] = (df.groupby(['Neighborhood'])['LotFrontage']
                     .transform(lambda x : x.fillna(x.median())))

df['Alley'] = df['Alley'].fillna('None')

df.loc[
    (df['MasVnrType'].isnull()) &
    (df['MasVnrArea'] == 0),
    'MasVnrType'
] = 'None'

df.loc[
    (df['MasVnrType'].isnull()) &
    (df['MasVnrArea'] != 0),
    'MasVnrType'
] = 'BrkFace'

df['BsmtQual'] = df['BsmtQual'].fillna('No Basement')

df['BsmtCond'] = df['BsmtCond'].fillna('No Basement')

df['BsmtFinType1'] = df['BsmtFinType1'].fillna('No Basement')


df.loc[
    (df['BsmtFinType2'].isnull()) &
    (df['TotalBsmtSF'] > 0),
    'BsmtFinType2'
] = df['BsmtFinType2'].mode()[0]
df['BsmtFinType2'] = df['BsmtFinType2'].fillna('No Basement')


df.loc[
    (df['BsmtExposure'].isnull()) &
    (df['TotalBsmtSF'] > 0),
    'BsmtExposure'
] = df['BsmtExposure'].mode()[0]
df['BsmtExposure'] = df['BsmtExposure'].fillna('No Basement')

df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

df['GarageFinish'] = df['GarageFinish'].fillna('No Garage')

df['GarageType'] = df['GarageType'].fillna('No Garage')

df['GarageQual'] = df['GarageQual'].fillna('No Garage')

df['GarageCond'] = df['GarageCond'].fillna('No Garage')

df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)

df['PoolQC'] = df['PoolQC'].fillna('No Pool')

df['Fence'] = df['Fence'].fillna('No Fence')

df['MiscFeature'] = df['MiscFeature'].fillna('None')

df['FireplaceQu'] = df['FireplaceQu'].fillna('No Fireplace')

df = df.drop(columns=['Id'])

# FEATURE CATEGORIZATION

categorical_features = [
    'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
    'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
    'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
    'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
    'CentralAir', 'Electrical', 'Functional', 'GarageType',
    'GarageFinish', 'PavedDrive', 'MiscFeature', 'SaleType',
    'SaleCondition'
]

ordinal_features = [
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
    'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond',
    'PoolQC', 'Fence'
]

# ENCODING

df = pd.get_dummies(data=df,columns=ordinal_features + categorical_features,drop_first=True)
df = df.astype(float)

# TRAIN TEST SPLIT

X = df.drop('SalePrice',axis=1)
Y = df['SalePrice']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)


# DECISION TREE

DTmodel = DecisionTreeRegressor(random_state=42)
DTmodel.fit(X_train,Y_train)

predictions1 = DTmodel.predict(X_test)

print("DecisionTree :",r2_score(Y_test,predictions1))

# RANDOM FOREST

rfmodel = RandomForestRegressor(random_state=42)
rfmodel.fit(X_train,Y_train)

predictions2 = rfmodel.predict(X_test)

print("RandomForest :",r2_score(Y_test,predictions2))

# FEATURE SELECTION

zero_importance_features = [
    'PoolQC_No Pool', 'MSSubClass_45', 'MSSubClass_75', 'MSSubClass_40',
    'MSSubClass_180', 'Street_Pave', 'PoolQC_Gd', 'PoolQC_Fa',
    'GarageCond_No Garage', 'GarageQual_No Garage', 'GarageQual_Po',
    'GarageCond_Gd', 'GarageCond_Po', 'LowQualFinSF', 'SaleType_CWD',
    'MiscFeature_TenC', 'MiscFeature_Othr', 'SaleType_Con',
    'SaleType_ConLw', 'SaleType_ConLI', 'SaleCondition_AdjLand',
    'SaleType_Oth', 'HeatingQC_Po', 'BsmtCond_Po',
    'Neighborhood_NPkVill', 'LandContour_Low', 'LotConfig_FR3',
    'Neighborhood_BrDale', 'LandSlope_Sev', 'Neighborhood_Blueste',
    'LotShape_IR3', 'Utilities_NoSeWa', 'RoofMatl_Metal',
    'RoofMatl_WdShngl', 'Exterior1st_CBlock', 'Exterior1st_BrkComm',
    'Exterior1st_AsphShn', 'Exterior1st_ImStucc', 'RoofStyle_Mansard',
    'RoofStyle_Shed', 'RoofMatl_Membran', 'Condition1_RRAn',
    'Condition2_Feedr', 'Condition1_RRNn', 'Condition1_RRNe',
    'Condition2_PosA', 'Condition2_RRNn', 'Condition2_PosN',
    'Condition2_RRAe', 'Condition2_RRAn', 'HouseStyle_2.5Fin',
    'HouseStyle_1.5Unf', 'HouseStyle_SFoyer', 'RoofStyle_Gambrel',
    'Exterior2nd_Stone', 'Exterior2nd_Other', 'Exterior2nd_CBlock',
    'Exterior2nd_BrkFace', 'Exterior1st_Stucco', 'Exterior1st_Stone',
    'Exterior2nd_AsphShn', 'Exterior2nd_Brk Cmn', 'RoofMatl_WdShake',
    'RoofMatl_Roll', 'BsmtFinType2_GLQ', 'ExterQual_Fa',
    'Foundation_Slab', 'Foundation_Wood', 'Foundation_Stone',
    'BsmtFinType1_No Basement', 'BsmtQual_No Basement', 'ExterCond_Po',
    'Electrical_FuseP', 'Functional_Maj2', 'Functional_Min1',
    'Heating_Wall', 'Heating_Grav', 'Heating_OthW',
    'BsmtCond_No Basement', 'BsmtFinType2_No Basement',
    'Heating_GasA', 'Electrical_Mix', 'Electrical_FuseF',
    'Functional_Sev', 'GarageFinish_No Garage',
    'GarageType_No Garage'
]

# REMOVE ZERO IMPORTANCE FEATURES

df = df.drop(columns=zero_importance_features)

# TRAIN TEST SPLIT AFTER FEATURE SELECTION

X = df.drop('SalePrice',axis=1)
Y = df['SalePrice']
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)

# XGBOOST

xg_model = XGBRegressor(max_depth=3,random_state=42)

# HYPERPARAMETER TUNING

param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [2, 3, 4, 5, 6],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5]
}

# RANDOMIZED SEARCH CV

random_search = RandomizedSearchCV(
    random_state=42,
    estimator=xg_model,
    n_iter=20,
    scoring='r2',
    n_jobs=-1,
    cv = 5,
    param_distributions=param_grid
)

random_search.fit(X_train,Y_train)

# FINAL MODEL EVALUATION

best_model = random_search.best_estimator_

print("Best Parameters:")
print(random_search.best_params_)

print("\nBest CV Score:")
print(random_search.best_score_)

print("\nTuned XGBoost Test R²:")
print(best_model.score(X_test, Y_test))




