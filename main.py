import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from collections import Counter

from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score


# Reads the data from the CSV file
df = pd.read_csv('spam.csv')

# Creates an instance of English stopwords
stop_words = stopwords.words('english')

# Cleans the text


def word_cleaner(message, stemming_flag=False, Lemmatisation_flag=False, stop_words=None):
    # Make all words lowercase
    message = str(message).lower()
    # Removes whitespaces
    message = str.strip(message)
    # Strips punctuation
    message = re.sub(r'[^\w\s]', '', message)
    # Tokenize(Converts String to List)
    message_lst = word_tokenize(message)
    # Remove Stopwords
    if message_lst is not None:
        message_lst = [word for word in message_lst if word not in stop_words]
    # Stemming (removing ing, ly, etc...)
    if stemming_flag:
        ps = PorterStemmer()
        message_lst = [ps.stem(word) for word in message_lst]
    # Lemmatisation (convert into root word)
    if Lemmatisation_flag:
        lem = WordNetLemmatizer()
        message_lst = [lem.lemmatize(word) for word in message_lst]
    # Converts back to sting from list
    message = " ".join(message_lst)
    return message

# New dataframe for the clean text


df['cleaned_text'] = df['Message'].apply(lambda message: word_cleaner(message, stop_words=stop_words))

# List of all the words found in Spam Emails
spam_corpus = []
for mail in df[df['Category'] == 'spam']['cleaned_text'].tolist():
    for word in mail.split():
        spam_corpus.append(word)

# List of all the words found in Ham Emails
ham_corpus = []
for mail in df[df['Category'] == 'ham']['cleaned_text'].tolist():
    for word in mail.split():
        ham_corpus.append(word)

# Creates a bar plot showing the 50 most common words in spam emails
# Delete the Quotes to show the bar plot

"""
plt.figure(figsize=(15, 8))
sns.barplot(x=pd.DataFrame(Counter(spam_corpus).most_common(30))[0], y=pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation=45)
plt.ylabel('Word Count')
plt.xlabel('Words')
plt.title('25 Most Common Words found in Spam Emails')
plt.show()
"""

# Assigns a 1 or 0 to ham or spam in the category dataframe
encoder = LabelEncoder()
encoder.fit(df['Category'])
df['Category'] = encoder.transform(df['Category'])
x = df['cleaned_text']
y = df['Category'].values

# Runs the If-Idf vectorizer of the cleaned text dataframe
vec = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
x = vec.fit_transform(x).toarray()

# Splits the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=42)

# MLP using SGD optimizer
opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
model = Sequential()
model.add(Dropout(0.3))
model.add(Dense(64, activation=tf.nn.relu))
model.add(Dropout(0.4))
model.add(Dense(32, activation=tf.nn.relu))
model.add(Dense(1, activation=tf.nn.sigmoid))
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=250, verbose=2)

# Gets the prediction of the test data using the model

prediction = model.predict(x_test)
prediction = prediction.flatten()
y_prediction = np.where(prediction >= 0.5, 1, 0)

# Creates a confusion matrix showing the True Ham, False Spam, False Ham, and True Spam

cf = confusion_matrix(y_test, y_prediction)
heatmap_names = ['True Ham', 'False Spam', 'False Ham', 'True Spam']
heatmap_count = ["{0:0.0f}".format(value) for value in cf.flatten()]
heatmap_percentage = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
heatmap_labels = [f"{var1}\n{var2}\n{var3}" for var1, var2, var3 in
                    zip(heatmap_names, heatmap_count, heatmap_percentage)]

heatmap_labels = np.asarray(heatmap_labels).reshape(2, 2)
sns.set(font_scale=1.4)
sns.heatmap(cf, annot=heatmap_labels, fmt='', cmap='Blues')
test_acc = accuracy_score(y_prediction, y_test)
plt.xlabel('\n Test Accuracy: ' + "{0:.2%}".format(test_acc))
test_precision = precision_score(y_test, y_prediction)
print('Test Accuracy: ' + "{0:.2%}".format(test_acc))
print('Model Precision : ' + "{0:.2%}".format(test_precision))
plt.show()

# Saves the model and all associated weights
# model.save("spamFilterModel")
