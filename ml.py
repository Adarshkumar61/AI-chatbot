# AI Chatbot Example
# We'll create a chatbot that can respond to basic user queries, such as:

# - Hello
# - How are you?
# - What's your name?
# - Tell me a joke

# We'll use the following tools:

# - Python as the programming language
# - NLTK (Natural Language Toolkit) for NLP tasks
# - scikit-learn for machine learning tasks

# Code

import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Importing the dataset:
train_data = [
    ("hello", "greeting"),
    ("hi", "greeting"),
    ("how are you?", "query"),
    ("what's your name?", "query"),
    ("tell me a joke", "joke"),
]

# Preprocess the data
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()

X = [lemmatizer.lemmatize(text) for text, _ in train_data]
y = [label for _, label in train_data]

X = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Define a function to respond to user queries
def respond(query):
    query = lemmatizer.lemmatize(query)
    query = vectorizer.transform([query])
    prediction = clf.predict(query)
    return prediction[0]

# Test the chatbot
print(respond("hello"))  # Output: greeting
print(respond("how are you?"))  # Output: query
print(respond("tell me a joke"))  # Output: joke
