# Disaster Response Pipeline Project

The Disaster response pipeline project was one of the projects I did for Udacity's
Nanodegree in Data Science program. The aim of the project was to build a
Natural Language Programming (NLP) pipeline that:
* Extract, Transforms and load the dataset
* Creating meaningful features from the data using tokenization and term-frequency times inverse document-frequency.
* Training a model on new data by performing grid search using a Multi-layer perceptron (MLP) algorithm
* Saving the model for later use
* Running a dashboard where the user can manually enter messages that will be classified

## Installation
### Dependencies

Disaster-response-pipeline was built using the following dependencies:
* Python 3.6.7
* Flask1.0.2
* nltk3.3
* pandas0.22.0
* plotly3.6.1
* scikit-learn0.19.1
* SQLAlchemy1.2.11

The requirements file can be used to install the dependencies, by running:

```
pip install -r requirements.txt
```

### User installation
Clone the github repository and install all the dependencies.

```
git clone git@github.com:Rmostert/Disaster-response-pipeline.git
```

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
