import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')

def accuracy_function(tp, tn, fp, fn):
    accuracy = (tp+tn) / (tp+tn+fp+fn)
    return accuracy


def precision_function(tp, fp):
    precision = tp / (tp+fp)
    return precision


def recall_function(tp, fn):
    recall = tp / (tp+fn)
    return recall

def confusion_matrix(truth, predicted):

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for true, pred in zip(truth, predicted):

        if true == '1' or 1:
            if pred == true:
                true_positive += 1
            elif pred != true:
                false_negative += 1

        elif true == '0' or 1:
            if pred == true:
                true_negative += 1
            elif pred != true:
                false_positive += 1

    accuracy = accuracy_function(true_positive, true_negative, false_positive, false_negative)
    precision = precision_function(true_positive, false_positive)
    recall = recall_function(true_positive, false_negative)

    return accuracy, precision, recall

def build_cnn():

    model = tf.keras.Sequential(name="CNN")

    model.add(tf.keras.layers.Input(shape=(4096, 1)))

    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=4, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    print(model.summary())

    return model

if __name__ == '__main__':
    df = pd.read_csv("./dataset/SQLiV3.csv")

    # Drop duplicate row
    df = df.drop_duplicates('Sentence')

    df = df.drop('Unnamed: 2', axis=1, errors='ignore')
    df = df.drop('Unnamed: 3', axis=1, errors='ignore')

    # Drop row with Nan
    df = df.dropna(how='any')

    # Drop row with incorrect label
    df = df[(df['Label'] == '0') | (df['Label'] == '1')]

    print(df.head())

    # Reset index
    df = df.sample(frac=1).reset_index(drop=True)
    y = np.array([int(i) for i in df['Label'].values])

    vectorizer = CountVectorizer(max_features=4096, ngram_range=(2, 2))
    #vectorizer = TfidfVectorizer(min_df=2, max_df=0.7, stop_words='english')

    count_matrix = vectorizer.fit_transform(df['Sentence']).toarray()

    X_train, X_test, y_train, y_test = train_test_split(count_matrix, y, test_size=0.2, random_state=42)

    # (sentence_index, feature_index) count
    print("Train shape: {}".format(X_train.shape))
    print("Test shape: {}".format(X_test.shape))

    # Logistic Regression
    model = LogisticRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy, precision, recall = confusion_matrix(y_test, y_pred)
    print("=================Logistic Regression Result=================")
    print("Accuracy : {:.4f}".format(accuracy))
    print("Precision : {:.4f}".format(precision))
    print("Recall : {:.4f}".format(recall))


    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy, precision, recall = confusion_matrix(y_test, y_pred)
    print("=================KNN Result=================")
    print("Accuracy : {:.4f}".format(accuracy))
    print("Precision : {:.4f}".format(precision))
    print("Recall : {:.4f}".format(recall))

    X_train = X_train.reshape(-1, 4096, 1)
    X_test = X_test.reshape(-1, 4096, 1)

    # CNN shape
    print("Train shape: {}".format(X_train.shape))
    print("Test shape: {}".format(X_test.shape))

    cnn = build_cnn()
    cnn.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4), metrics=['accuracy'])
    cnn.fit(X_train, y_train, batch_size=10, epochs=5, validation_data=(X_test, y_test))

    y_pred = cnn.predict(X_test)

    # Turn sigmoid result to 0 or 1
    y_pred = np.where(y_pred > 0.5, 1, 0)
    
    accuracy, precision, recall = confusion_matrix(y_test, y_pred)
    print("=================CNN Result=================")
    print("Accuracy : {:.4f}".format(accuracy))
    print("Precision : {:.4f}".format(precision))
    print("Recall : {:.4f}".format(recall))