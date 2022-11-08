import numpy as np
import pandas as pd
import pickle as pk

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
)

# import packages for hyperparameters tuning
import xgboost as xgb

# Import the data
print(f"Reading the data ...!")
df = pd.read_csv("./final_data.csv")

# Some records are not unique
df.drop_duplicates(inplace=True)

# The target feature is `arrest` which is a boolean feature. For the medel we need to change this to `0` and `1` values.
# We will also change the `domestic` feature in the same manner

df.arrest = df.arrest.astype(int)
df.domestic = df.domestic.astype(int)

# Identify Catergorical and Numeric Columns

categorical_columns = [
    "iucr",
    "primary_type",
    "description",
    "location_description",
    "fbi_code",
    "zip",
    "street",
]

numerical_columns = [
    "domestic",
    "beat",
    "district",
    "ward",
    "community_area",
    "latitude",
    "longitude",
    "hour",
    "day",
]

features = categorical_columns + numerical_columns

# # XGBoost
# Split the Data
# The data will be split as follows:
# 
# Training 
# 80% of the data will be used for Training.
# 
# Test
# 20% of the data will be help back for final testing of the model.

df_train, df_test = train_test_split(df, test_size=0.2, random_state=11)

# Use a dictionay vertorizer
dict_train = df_train[features].to_dict(orient="records")
dict_test = df_test[features].to_dict(orient="records")

dv = DictVectorizer(sparse=False)

# Transform the feature data
X_train = dv.fit_transform(dict_train)
X_test = dv.transform(dict_test)

# The target values
y_train = df_train.arrest.values
y_test = df_test.arrest.values

# XGB parameters
# HyperOpt was used to tune the XGBoost paramerters
# http://hyperopt.github.io/hyperopt/#hyperopt-distributed-asynchronous-hyper-parameter-optimization


colsample_bytree = 0.4
learning_rate = 0.2
max_depth = 8
min_child_weight = 6
subsample = 0.8899372509684718


# Train the final model
print("Training the model ...!")
xg = xgb.XGBClassifier(
    colsample_bytree=colsample_bytree,
    learning_rate=learning_rate,
    max_depth=max_depth,
    min_child_weight=min_child_weight,
    subsample=0.8223746574068592,
    eval_metric="rmse",
    early_stopping_rounds=10,
)

evaluation = [(X_train, y_train), (X_test, y_test)]

xg.fit(
    X_train,
    y_train,
    eval_set=evaluation,
    verbose=False,
)

y_pred = xg.predict(X_test)

print(f"Model Metrics ...!\n")
print(f"Confusion Matrix Tree: \n")
print(confusion_matrix(y_test, y_pred), "\n")
print(f"The precision for Tree is: {precision_score(y_test, y_pred)}")
print(f"The recall for Tree is: {recall_score(y_test, y_pred)}")
print()

print(f"Exporting the Model and the DictVectorizer")

model_file = "model.bin"
fout = open(model_file, "wb")
pk.dump(xg, fout)
fout.close()

dv_file = "dv.bin"
fout = open(dv_file, "wb")
pk.dump(dv, fout)
fout.close()