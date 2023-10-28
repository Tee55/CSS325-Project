from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
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

df = pd.read_csv("./dataset/SQLiV3.csv")

# Drop duplicate row
df = df.drop_duplicates('Sentence')

df = df.drop('Unnamed: 2', axis=1, errors='ignore')
df = df.drop('Unnamed: 3', axis=1, errors='ignore')

# Drop row with Nan
df = df.dropna(how='any')

print(df)

# Reset index
df = df.sample(frac=1).reset_index(drop=True)
y = df['Label'].values

vectorizer = CountVectorizer(ngram_range=(2, 2))
#vectorizer = TfidfVectorizer(min_df=2, max_df=0.7, stop_words='english')

count_matrix = vectorizer.fit_transform(df['Sentence'])
for name in vectorizer.get_feature_names_out():
    print(name)

X_train, X_test, y_train, y_test = train_test_split(count_matrix, y, test_size=0.2, random_state=42)

# (sentence_index, feature_index) count
print(X_train.shape, X_test.shape)
print(X_train)

# Linear Regression
model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy, precision, recall = confusion_matrix(y_test, y_pred)
print("Accuracy : {:.4f}".format(accuracy))
print("Precision : {:.4f}".format(precision))
print("Recall : {:.4f}".format(recall))
