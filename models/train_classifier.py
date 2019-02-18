#Import libraries
import sys
from sqlalchemy import create_engine
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import re

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.metrics import (classification_report,precision_score,
                             recall_score, f1_score, make_scorer)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
import pickle


def load_data(database_filepath):
    '''
    Function to read a SQLlite table into a pandas dataframe

    Arguments:
        database_filepath: Location to the SQLlite database
    Return:
        X: A pandas dataframe containing the messages
        Y: A pandas dataframe containing the ourcome variables
        category_names: a list containing the categories

    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Disaster_messages',engine )
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns

    drop_cols = category_names[Y.max() == 0]
    Y = Y.drop(drop_cols,axis=1)



    return X, Y, category_names


def tokenize(text):
    '''
    Function to clean and tokenize disaster messages_filepath

    Arguments:
        text: string. The message that needs to be tokenized

    Return:
        clean_tokens: A list of tokens
    '''


    tokens = word_tokenize(re.sub(r"[^a-zA-Z0-9]",' ',text.lower().strip()))
    lemmatizer = WordNetLemmatizer()
    stopWords = set(stopwords.words('english'))

    wordsFiltered = []

    for w in tokens:
        if w not in stopWords:
            wordsFiltered.append(w)


    clean_tokens = []
    for tok in wordsFiltered:
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():

    '''
    Function for initialising the model Pipeline and doing Gridsearch

    Argument:
        None
    Return:
        A scikit-learn model object

    '''
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MLPClassifier())
                 ])

    param_grid = {'clf__hidden_layer_sizes': [100,200],
              'clf__learning_rate': ['constant', 'invscaling'],
              'clf__solver': ['adam','sgd']
              }

    f1_scorer = make_scorer(f1_score,average='weighted')
grid_search = GridSearchCV(pipeline,param_grid=param_grid,scoring=f1_scorer,
                cv=5,n_jobs=-1)

    return grid_search


def evaluate_model(model, X_test, Y_test, category_names):

    '''
    A function for evaluating the fitted model on the test datasets

    Arguments:
        model: A scikit-learn model object
        X_test: A pandas dataframe containing the messages in the test dataset
        Y_test: A pandas dataframe containing the responses in the test dataset
        category_names: A list contaiining the outcome categories

    Return:
        None
    '''

    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names = category_names))



def save_model(model, model_filepath):

    '''
    A function to save the fitted model as a pickle file

    Arguments:
    model: a scikit-learn model object
    model_filepath: the location where the pickle file needs to be save_model

    Return:
        None

    '''


    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
