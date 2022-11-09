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

The development and deployment environments assume:

* Python 3.10
* Docker

Are both installed and properly running on you machine. Installing and confirguring Python and Docker are beyond the scope of this read me.

This project uses `pipenv`. If pipenv is not installed on your machine you can install it by running:

> `pip install pipenv`

The required libraries can be installed by running:

> `pipenv install`

Using `pipenv graph` we can see the required libraries and versions (less `jupyter` and `jupyterlab` which are included in the development environment to allow the notebooks to be executed):

```
git/github/mlzoomcamp-midterm-project  main ✗                                                                                   21m ⚑
▶ pipenv graph
Flask==2.2.2
  - click [required: >=8.0, installed: 8.1.3]
  - itsdangerous [required: >=2.0, installed: 2.1.2]
  - Jinja2 [required: >=3.0, installed: 3.1.2]
    - MarkupSafe [required: >=2.0, installed: 2.1.1]
  - Werkzeug [required: >=2.2.2, installed: 2.2.2]
    - MarkupSafe [required: >=2.1.1, installed: 2.1.1]
folium==0.13.0
  - branca [required: >=0.3.0, installed: 0.5.0]
    - jinja2 [required: Any, installed: 3.1.2]
      - MarkupSafe [required: >=2.0, installed: 2.1.1]
  - jinja2 [required: >=2.9, installed: 3.1.2]
    - MarkupSafe [required: >=2.0, installed: 2.1.1]
  - numpy [required: Any, installed: 1.23.4]
  - requests [required: Any, installed: 2.28.1]
    - certifi [required: >=2017.4.17, installed: 2022.9.24]
    - charset-normalizer [required: >=2,<3, installed: 2.1.1]
    - idna [required: >=2.5,<4, installed: 3.4]
    - urllib3 [required: >=1.21.1,<1.27, installed: 1.26.12]
gunicorn==20.1.0
  - setuptools [required: >=3.0, installed: 65.5.1]
hyperopt==0.2.7
  - cloudpickle [required: Any, installed: 2.2.0]
  - future [required: Any, installed: 0.18.2]
  - networkx [required: >=2.2, installed: 2.8.8]
  - numpy [required: Any, installed: 1.23.4]
  - py4j [required: Any, installed: 0.10.9.7]
  - scipy [required: Any, installed: 1.9.3]
    - numpy [required: >=1.18.5,<1.26.0, installed: 1.23.4]
  - six [required: Any, installed: 1.16.0]
  - tqdm [required: Any, installed: 4.64.1]
pandas==1.5.1
  - numpy [required: >=1.21.0, installed: 1.23.4]
  - python-dateutil [required: >=2.8.1, installed: 2.8.2]
    - six [required: >=1.5, installed: 1.16.0]
  - pytz [required: >=2020.1, installed: 2022.6]
scikit-learn==1.1.3
  - joblib [required: >=1.0.0, installed: 1.2.0]
  - numpy [required: >=1.17.3, installed: 1.23.4]
  - scipy [required: >=1.3.2, installed: 1.9.3]
    - numpy [required: >=1.18.5,<1.26.0, installed: 1.23.4]
  - threadpoolctl [required: >=2.0.0, installed: 3.1.0]
xgboost==1.7.1
  - numpy [required: Any, installed: 1.23.4]
  - scipy [required: Any, installed: 1.9.3]
    - numpy [required: >=1.18.5,<1.26.0, installed: 1.23.4]
```

Note: If the Folium maps are not displayed you need to trust the notebook:

1. Open a terminal
2. Navigate to the project folder
3. Type:
   1. `jupyter trust 01_eda.ipynb`



# Export Training Logic to Script

XGBoost was chose as the best model. The Python in the XGBoost notebook was export and cleaned up.

Logic was then also added to export the model and the dictionary vectorizer to a binany file using the Pickle library.

See [generate_model.py](./generate_model.py).

To generate the model and dictionary vectoriser binary files open a command prompt in your environment (setup instructions above) and type:

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

The simplest way to get the code for this Github repository is to:

1.  Navigate to the project [Guthub](https://github.com/davidcolton/mlzoomcamp-midterm-project) page
2. Click the `Code` button
3. Download a Zip of the code
4. Unzip the code to a location of your choice on you laptop.

Once you have the code and the environment setup you will be able to run all stages of the project from notebooks to Docker files.

# Model Deployment

To deploy the model locally Flask is used. The Flask server can be run using [flask_serv.py](./flask_serv.py):

1. Open a command prompt
2. Navigate to the project folder
3. Type `python flask_serv.py`

You should see:

```
python flask_serv.py
 * Serving Flask app 'arrest'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:9696
 * Running on http://192.168.68.73:9696
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 229-968-480
```

The model is now ready to accept requests.

The model can be tested using [flask_test.py](./flask_test.py):

1. Open a second command prompt
2. Navigate to the project folder
3. Type `python flask_test.py`

You should see:

```
git/github/mlzoomcamp-midterm-project  main ✗                                                                                   21m ⚑
▶ python flask_test.py
{'arrest': False, 'arrest_probability': 0.07704198360443115}
```

This shows that the probability of the sample crime details are unlikely to lead to an arrest.



# Containerisation

The supplied `Dockerfile`can be built and deployed using the following commands:

1. Start Docker in your environment
2. Build the Docker image
    - `docker build -t arrest .`
3. Run the Docker image
    - `docker run -it --rm -p 9696:9696 arrest`
4. Test the Docker image using the given Python test file as above
    - `python flask_test.py  `



# Cloud Deployment

Not attempted
