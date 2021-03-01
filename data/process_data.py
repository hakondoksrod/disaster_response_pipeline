import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''Load messages and categories data and merge'''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    '''Clean the merged data and return dataframe'''
    
    #Create dataframe with the 'categories' column to split and clean
    cat_split = df.categories.str.split(';', expand=True)
    cat_split.columns = cat_split.iloc[0].str[:-2].tolist()
    
    #Remove everything before the '-' in the category columns, leaving the category name
    for col in cat_split.columns:
        cat_split[col] = cat_split[col].apply(lambda x: x[-1:])
        cat_split[col] = cat_split[col].astype(int)
    
    #Change values '2' to '1' in the 'related' column
    cat_split.loc[cat_split['related'] == 2] = 1
    
    #Merge with original dataframe and drop original cateogries column
    df = pd.concat([df, cat_split], axis=1)
    df.drop('categories', axis=1, inplace=True)
    
    #Drop duplicated data
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    '''Save the cleaned data to an sqlite database'''
    
    conn = f"sqlite:///{database_filename}"
    engine = create_engine(conn)
    df.to_sql('disaster_response', engine, index=False, if_exists='replace')  


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