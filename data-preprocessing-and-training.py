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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
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


def select_data(dataframe):
    # Filter the dataset minimum having 100 rows per category
    dataframe = dataframe.groupby("CATEGORY").filter(lambda x: len(x) >= 1)
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
    return tfidf_dataset.toarray(), class_label, preprocessor, le


def train_decision_tree(X_train, y_train):
    # Create a Decision Tree classifier
    dt_model = DecisionTreeClassifier(criterion="gini", splitter="best", random_state=42)

    # Train the classifier on the training dataset
    dt_model.fit(X_train, y_train)

    return dt_model


def train_naive_bayes(X_train, y_train):
    # Define the naive bayes classifier
    NBClass = MultinomialNB()

    # Train the final model on the entire training set
    NBClass.fit(X_train, y_train)

    return NBClass


def train_neural_network(X_train, y_train, num_classes, epochs=20, batch_size=64):
    # Build the neural network model
    model = Sequential()
    # Input layer
    model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
    # Hidden layers
    model.add(Dense(32, activation='relu'))
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    return model


def evaluate_model(model, y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=1)

    print(f"Accuracy score of {model}: {accuracy}")
    print(f"Precision score of {model}: {precision}")
    print(f"Recall score of {model}: {recall}")


def predict_category(model, preprocessor, le, title):
    # Tokenize and preprocess the input title
    input_data = preprocessor.transform([title]).toarray()

    # Make prediction using the specified model
    if isinstance(model, DecisionTreeClassifier) or isinstance(model, MultinomialNB):
        # Decision Tree and Naive Bayes return labels directly
        prediction = model.predict(input_data)
        # Get the probabilities for each class
        probabilities = model.predict_proba(input_data)[0]
    elif isinstance(model, Sequential):
        # Neural Network returns probabilities, find the class with the highest probability
        prediction = np.argmax(model.predict(input_data), axis=-1)
        # Get the probabilities for each class
    else:
        raise ValueError("Unsupported model type")

    # Inverse transform the label to get the original category
    category = le.inverse_transform(prediction)
    print(f"Predicted category: {prediction}")

    return category


# Main pipeline
data = load_data()
selected_data = select_data(data)
cleaned_data = clean_data(selected_data)
X, y, preprocessor, le = preprocess_data(cleaned_data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree
dt_model = train_decision_tree(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
evaluate_model("Decision Tree", y_pred_dt, y_test)

# Train Naive Bayes
nb_model = train_naive_bayes(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
evaluate_model("Naive Bayes", y_pred_nb, y_test)

# Train Neural Network
num_classes = len(np.unique(y))
nn_model = train_neural_network(X_train, y_train, num_classes)
y_pred_nn = nn_model.predict(X_test)
# evaluate_model("Neural Network", y_pred_nn, y_test)

# Predict category
input_title = "airport accident Alabama Airline"

predicted_category_dt = predict_category(dt_model, preprocessor, le, input_title)
predicted_category_nb = predict_category(nb_model, preprocessor, le, input_title)
predicted_category_nn = predict_category(nn_model, preprocessor, le, input_title)

print(f"Predicted category (Decision Tree): {predicted_category_dt}")
print(f"Predicted category (Naive Bayes): {predicted_category_nb}")
print(f"Predicted category (Neural Network): {predicted_category_nn}")