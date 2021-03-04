import sys
from sqlalchemy import create_engine
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import pickle
import re
import nltk
nltk.download(['punkt', 'wordnet'])

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import PassiveAggressiveClassifier


def load_data(database_filepath):
    '''Load data from the SQLite database file and divide into X and y'''
    conn = f"sqlite:///{database_filepath}"
    engine = create_engine(conn)
    df = pd.read_sql_table('disaster_response', engine)
    X = df['message'].values
    y = df.iloc[:,4:]
    features = y.columns.tolist()

    return X, y, features


def tokenize(text):
    '''
    Function to tokenize and lemmatize text input.

    Input:
        text: data for tokenization and lemmatization
    Output:
        clean_tokens: list of clean tokens
    '''
    #Remove all punctuation from the text
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text)

    #Tokenize and lemmatize the text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''Build model using machine learning pipeline'''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(PassiveAggressiveClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 1.0),
        #'vect__max_features': (None, 5000),
        #'tfidf__use_idf': (True, False),
        'clf__estimator__C': (0.5,1),
        'clf__estimator__tol': (0.001, 0.002),
        'clf__estimator__max_iter': (1000, 1500),
    }

    cv = GridSearchCV(pipeline, param_grid = parameters, n_jobs=-1, verbose=2, cv=3)

    return cv


def evaluate_model(model, X_test, y_test, features):
    '''Display classification report'''
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=features))


def save_model(model, model_filepath):
    '''Save model as a pickle file'''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, features = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, features)

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
