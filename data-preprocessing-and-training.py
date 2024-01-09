# Importing necessary libraries
import re
import string

import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import set_config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def load_data(src_file='sample_dataset.csv'):
    # Load data from a CSV file into a Pandas DataFrame
    dataframe = pd.read_csv(src_file, encoding="utf8", quotechar="\"", engine='python', usecols=["TITLE", "CATEGORY"])
    return dataframe


def clean_data(dataframe):
    # Check for missing data
    if any(dataframe.isnull().any()):
        print('Missing Data\n')
        print(dataframe.isnull().sum())
    else:
        print('No missing data')

    # Check for duplicate rows
    if any(dataframe.duplicated()):
        print('Duplicate rows found')
        print('Number of duplicate rows= ', dataframe[dataframe.duplicated()].shape[0])
        # Drop duplicate rows and reset index
        dataframe.drop_duplicates(inplace=True, keep='first')
        dataframe.reset_index(inplace=True, drop=True)
        print('Dropping duplicates\n')
        print(dataframe.shape)
    else:
        print('No duplicate data')

    return dataframe


def preprocess_data(dataframe):
    # Set sklearn config to output Pandas DataFrame
    set_config(transform_output="pandas")
    wnl = WordNetLemmatizer()

    # Tokenization and lemmatization function
    def tokenize(doc):
        document = doc.lower()
        document = re.sub(r'\d+', '', document)
        document = document.translate(str.maketrans('', '', string.punctuation))
        document = document.strip()
        return [wnl.lemmatize(token) for token in word_tokenize(document) if token not in stopwords.words('english')]

    # Create a pipeline for text preprocessing
    preprocessor = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize, token_pattern=None)),
    ])

    # Transform the text data using TF-IDF
    tfidf_dataset = preprocessor.fit_transform(dataframe["TITLE"].values)

    # Encode the target labels
    le = LabelEncoder()
    class_label = le.fit_transform(dataframe["CATEGORY"])

    # Convert TF-IDF sparse matrix to dense array
    return tfidf_dataset.toarray(), class_label


def train_decision_tree(X_train, y_train):
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30, 40, 50],
    }

    # Create a decision tree classifier
    DTClass = DecisionTreeClassifier(random_state=42)

    # Use GridSearchCV with cross-validation to find the best hyperparameters
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(DTClass, param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best model from the grid search
    best_DTClass = grid_search.best_estimator_

    print("Best Decision Tree Hyperparameters:", grid_search.best_params_)
    print("Decision Tree Cross-Validation Mean Accuracy:", grid_search.best_score_)

    return best_DTClass


def train_naive_bayes(X_train, y_train):
    # Define the naive bayes classifier
    NBClass = MultinomialNB()

    # Use cross-validation to evaluate the model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(NBClass, X_train, y_train, cv=cv, scoring='accuracy')

    # Train the final model on the entire training set
    NBClass.fit(X_train, y_train)

    print("Naive Bayes Cross-Validation Scores:", scores)
    print("Mean Accuracy:", np.mean(scores))

    return NBClass


def train_neural_network(X_train, y_train, num_classes, epochs=20, batch_size=64):
    # Train a neural network model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    return model


def evaluate_model(model, X_test, y_test):
    # Initialize y_pred with None
    y_pred = None

    # Evaluate the performance of the model
    if isinstance(model, (DecisionTreeClassifier, MultinomialNB)):
        y_pred = model.predict(X_test)
    elif isinstance(model, Sequential):
        y_pred = np.argmax(model.predict(X_test), axis=-1)

    if y_pred is not None:
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=1)

        print(f"Accuracy score of {model.__class__.__name__}: {accuracy}")
        print(f"Precision score of {model.__class__.__name__}: {precision}")
        print(f"Recall score of {model.__class__.__name__}: {recall}")
        print(f"F1 score of {model.__class__.__name__}: {f1}")
    else:
        print("Error: y_pred is not initialized.")


# Main pipeline
data = load_data()
cleaned_data = clean_data(data)
X, y = preprocess_data(cleaned_data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree
dt_model = train_decision_tree(X_train, y_train)
evaluate_model(dt_model, X_test, y_test)

# Train Naive Bayes
nb_model = train_naive_bayes(X_train, y_train)
evaluate_model(nb_model, X_test, y_test)

# Train Neural Network
num_classes = len(np.unique(y))
nn_model = train_neural_network(X_train, y_train, num_classes)
evaluate_model(nn_model, X_test, y_test)
