#Import libraries

import sys
import pandas as pd
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    '''
    Function for loading disaster reponse messages_filepath
    Arguments:
        messages_filepath: File path to file containing disaster
                           response messages
        categories_filepath: File path to file containing disaster
                             response classification

    Returns:
        df: A dataframe containing the merged datasets

    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(left=messages,right=categories,how='left',on='id')
    return df


def clean_data(df):
    '''
    Function for cleaning the disaster response message dataset
    Arguments:
        df: Pandas dataframe


    Returns:
        df: Pandas dataframe

    '''

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda row: row[:-2]).values

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda row: row[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    df.drop('categories',axis=1,inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    # Remove erroneous values
    df = df[df['related']!=2]

    # drop duplicates
    df_dedup = df.drop_duplicates(subset='id')
    df = df_dedup.drop_duplicates(subset='message',keep=False)

    return df


def save_data(df, database_filename):
    '''
    Function for saving a dataset to a SQLlite database
    Arguments:
        df: Pandas dataframe. Dataset that needs to be saved
        database_filename: Location where database should be saved

    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Disaster_messages', engine, index=False,if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
