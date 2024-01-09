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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Reading and cleaning the dataset
src_file = 'dataset.csv'
dataframe = pd.read_csv(src_file, encoding="utf8", quotechar="\"", engine='python', usecols=["TITLE", "CATEGORY"])

# Data Cleaning
# Check for missing data
if any(dataframe.isnull().any()):
    print('Missing Data\n')
    print(dataframe.isnull().sum())
    dataframe["TITLE"].fillna("missing", inplace=True)  # Replace NaN with "missing" for example
else:
    print('No missing data')

# Check for duplicate rows
if any(dataframe.duplicated()):
    print('Duplicate rows found')
    print('Number of duplicate rows= ', dataframe[dataframe.duplicated()].shape[0])
    dataframe.drop_duplicates(inplace=True, keep='first')
    dataframe.reset_index(inplace=True, drop=True)
    print('Dropping duplicates\n')
    print(dataframe.shape)
else:
    print('No duplicate data')

# Data Transformation
set_config(transform_output="pandas")

wnl = WordNetLemmatizer()


# Function for cleaning and tokenizing the headline
def tokenize(doc):
    # Convert to lowercase, remove numbers, punctuation, and leading/trailing whitespaces
    document = doc.lower()
    document = re.sub(r'\d+', '', document)
    document = document.translate(str.maketrans('', '', string.punctuation))
    document = document.strip()
    # Lemmatize and remove stopwords
    return [wnl.lemmatize(token) for token in word_tokenize(document) if token not in stopwords.words('english')]


# Preprocessing Pipeline
preprocessor = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize, stop_words='english', token_pattern=None)),
])

# Transforming the dataset using TF-IDF
tfidf_dataset = preprocessor.fit_transform(dataframe["TITLE"].values)

# Label Encoding
le = LabelEncoder()
class_label = le.fit_transform(dataframe["CATEGORY"])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    tfidf_dataset.toarray(),
    class_label,
    test_size=0.3,
    random_state=42
)

# Decision Tree Classifier
DTClass = DecisionTreeClassifier(criterion="gini", splitter="best", random_state=42)
DTClass.fit(X_train, y_train)
y_pred_dt = DTClass.predict(X_test)

# Evaluating Decision Tree Model
print("Accuracy score of Decision Tree:", accuracy_score(y_test, y_pred_dt))
print("Precision score of Decision Tree:", precision_score(y_test, y_pred_dt, average='weighted', zero_division=1))
print("Recall score of Decision Tree:", recall_score(y_test, y_pred_dt, average='weighted', zero_division=1))
print("F1 score of Decision Tree:", f1_score(y_test, y_pred_dt, average='weighted', zero_division=1))

# Naive Bayes Classifier
NBClass = MultinomialNB()
NBClass.fit(X_train, y_train)
y_pred_nb = NBClass.predict(X_test)

# Evaluating Naive Bayes Model
print("\nAccuracy score of Naive Bayes:", accuracy_score(y_test, y_pred_nb))
print("Precision score of Naive Bayes:", precision_score(y_test, y_pred_nb, average='weighted', zero_division=1))
print("Recall score of Naive Bayes:", recall_score(y_test, y_pred_nb, average='weighted', zero_division=1))
print("F1 score of Naive Bayes:", f1_score(y_test, y_pred_nb, average='weighted', zero_division=1))

# Neural Network Model
num_classes = len(np.unique(class_label))

# Neural Network Model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1)

# Predictions for Neural Network Model
y_pred_nn = np.argmax(model.predict(X_test), axis=-1)

# Evaluating Neural Network Model
print("\nAccuracy score of Neural Network:", accuracy_score(y_test, y_pred_nn))
print("Precision score of Neural Network:", precision_score(y_test, y_pred_nn, average='weighted', zero_division=1))
print("Recall score of Neural Network:", recall_score(y_test, y_pred_nn, average='weighted', zero_division=1))
print("F1 score of Neural Network:", f1_score(y_test, y_pred_nn, average='weighted', zero_division=1))
