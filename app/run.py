import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index.str.title())
    genre_related = df[df['related'] == 1].groupby('genre').count()['message']

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    name='Total Messages'
                ),
                Bar(
                    x=genre_names,
                    y=genre_related,
                    name='Related Messages'
                )
            ],

            'layout': {
                'title': {
                    'text': '<b>Distribution of Message Genres</b><br>'
                    'Total messages and related (relevant) messages',
                    'font': {'size': 16}
                },
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode': "group"
            }
        },
        {
            'data': [
                Bar(
                    x=df.iloc[:,4:].sum().sort_values(ascending=True).values.tolist(),
                    y=df.iloc[:,4:].sum().sort_values(ascending=True).index.str.replace('_', ' ').str.title().tolist(),
                    orientation='h'
                )
            ],

            'layout': {
                'title': {
                    'text': '<b>Distribution of Message Categories</b>',
                    'font': {'size': 16}
                },
                'yaxis': {
                    'dtick': 1
                },
                'xaxis': {
                    'title': 'Count'
                },
                'height': 600,
                'margin': {
                    'r': 40,
                    'l': 150,
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
