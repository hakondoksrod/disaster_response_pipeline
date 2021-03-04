# Disaster Response Classification Web App
This web app uses a text classification machine learning model to categorize messages received during a humanitarian or environmental crisis. The model is trained on a data set containing abuot 30,000 real messages from various natural disasters. These messages have been encoded with 36 different categories depending on the content of the message. The data can be found [here](https://appen.com/datasets/combined-disaster-response-data/)

## Table of Contents:

## Installation:
Follow these intructions to run a copy of this web app on your local machine.

### Prerequisites
The full prerequisites for running the code is contained in the 'requirements.txt' file. For easy execution you could set up a virtual environemnt on your local machine and run the following code:
```cli
pip install -r requirements.txt
```
The main libraries used for this project are:
- Flask (web application framework)
- Plotly (data visualization)
- NLTK (natural language processing)
- Scikit-learn (Python machine learning library)
- Pandas (Python data analysis library)
- SQLalchemy (SQL database engine)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        ```cli
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        ```
    - To run ML pipeline that trains classifier and saves
        ```cli
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        ```

2. Run the following command in the project's root directory to run your web app.
    ```cli
    python app/run.py
    ```

3. Go to http://0.0.0.0:3001/

## File descriptions
- /app - folder containing files needed to run the web app
- /data
    - process_data.py - python script to load, clean and save the data as an SQL database file
- /model
    - train_classifier.py - python script to train and save the model as a pickle file

## Project Summary
This project is made as a part of the Udacity Data Scientist Nanodegree program. Using a dataset containing 30,000 categorized messages from real natural and humanitarian disasters, the project uses natural language processing and machine learning to classify messages into categories. One of the goals of the project was to set up an ETL pipeline (Extract, Transform, Load) for the dataset in order to load, clean and prepare the data for machine learning, and a machine learning pipeline for training and fitting the model.

I experimented with several different machine learning algorithms before deciding on a model. In the end, the most promising candidates were Linear SVC and PassiveAggressiveClassifier. The metrics for scoring the model were precision, recall and F1 score, which conveys the balance between precision and recall. Both of these algorithms provided the best F1 score among the tested algorithms, and they were both among the fastest to run, which is also important when running Grid Search to identify the best hypterparameters.

The imbalanced nature of the dataset became the deciding factor for choice of algorithm. Some categories contained relatively few messages, which impacted the model's ability to train on data from these categories. For messages from these categories, recall becomes very important. Seeing as the nature of these messages are very important, we would rather have false positives than false negatives, to minimize the risk of missing out on important messages. Since recall penalizes false negatives, I wanted to be extra aware of the recall score. 

Ultimately, both Linear SVC and PassiveAggressive Classifier yielded more or less the same F1 score, but PassiveAggressive performed a bit better on recall. The final scores after running a GridSearch to tune hyperparameters were a precision of 0.72, recall of 0.61 and an F1 score of 0.64. These are the weighted average scores of each category.

## Screenshots
