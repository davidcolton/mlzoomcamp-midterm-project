# mlzoomcamp-midterm-project

This is my Repositry for the ML Zoom Camp Midterm Project.

# Problem Description

The goal of this project is to determine if we can predict if an arrest will be made from the data in the Chicago Police Department's [CLEAR](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present-Dashboard/5cd6-ry5g) (Citizen Law Enforcement Analysis and Reporting) system. This dataset reflects reported incidents of crime (with the exception of murders where data exists for each victim) that occurred in the City of Chicago in the last year, minus the most recent seven days. In order to protect the privacy of crime victims, addresses are shown at the block level only and specific locations are not identified.

The project is broken down into several separate Notebooks as follows:

# Exploratory Data Analysis

[Data](./01_eda.ipynb)

The Data notebook imports the raw data and does some initial eploratory data analysis. It imports the raw `raw_data.csv` file and, on completion of analysis, saves the prepared data to a `training_data.csv` file. 



# Model Training

[Decision Tree](./02_decision_tree.ipynb)

In the Decision Tree notebook we create a decision tree model using area under the curve to measure it the best model. The best parameters for each model are determine using a K-Fold cross validation and Grid Search mechanism.

The data is split into a 80% training and 20% testing split. The training data is further split 80% / 20% for K-Fold cross validation.

The following parameters are optomised:

1. Max depth
2. Minimum samples leaf
3. Minimun sample split

Once the best parameters are chosen the accuracy of the model is tested using the held back test data using a confusion matrix, precision and recall.

[Random Forest](./03_random_forest.ipynb)

In the Random Forest notebook we create a random model using accuracy to measure it the best model. The best parameters for each model are determine using a K-Fold cross validation and a Randomised Search mechanism.

The data is split into a 80% training and 20% testing split. The training data is further split 80% / 20% for K-Fold cross validation.

The following parameters are optomised:

1. Number of Estimators
2. Minimum samples leaf
3. Minimun sample split
4. Maximum depth
5. Bootstrap

Once the best parameters are chosen the accuracy of the model is tested using the held back test data using a confusion matrix, precision and recall.

Looking at the confusion matrix and the value for recall we can see that this model is worse than the decision tree model.

[XGBoost](./04_xgboost.ipynb)

In the XGBoost notebook we creating a XGBoost model using root mean squared error to measure the best model. The best parameters for each model are determine using the Hyper Opt Library.

The data is split into a 80% training and 20% testing split. 

The following parameters are optomised:

1. Learning Rate
2. Maximum depth
3. Minimum child weight
4. Column sample by tree
5. Sub sample ratio

Once the best parameters are chosen the accuracy of the model is tested using the held back test data using a confusion matrix, precision and recall.



# Environment Management





# Export Training Logic to Script

XGBoost was chose as the best model. The Python in the XGBoost notebook was export and cleaned up.

Logic was then also added to export the model and the dictionary vectorizer to a binany file using the Pickle library.

See [generate_model.py](./generate_model.py).

To generate the model and dictionary vectoriser binary files open a command prompt in your environment (setup instructions below) and type:

> `python generate_model.py`

The output should look something like:

```
▶ python generate_model.py
Reading the data ...!
Training the model ...!
Model Metrics ...!

Confusion Matrix Tree:

[[11116   169]
 [ 1347  1742]]

The precision for Tree is: 0.9115646258503401
The recall for Tree is: 0.5639365490449983

Exporting the Model and the DictVectorizer
```



# Reproducibility





# Model Deployment





# Containerisation

The supplied `Dockerfile`can be built and deployed using the following commands:

```
1. Start Docker in your environment
2. Build the Docker image
    - docker build -t arrest .
3. Run the Docker image
    - docker run -it --rm -p 9696:9696 arrest
4. Test the Docker image using the given Python test file
    - python flask_test.py   
```



# Cloud Deployment

Not attempted
