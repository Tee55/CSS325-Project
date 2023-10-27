import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

df = pd.read_csv("./dataset/SQLiV3.csv")
df = df.drop_duplicates('Sentence')
df = df.drop('Unnamed: 2', axis=1)
df = df.drop('Unnamed: 3', axis=1)
df = df.dropna(how='any')

train = df.sample(frac=1).reset_index(drop=True)

train = df.iloc[:30000]
test = df.iloc[30000:]

from sklearn.feature_extraction.text import CountVectorizer

countVectorizer = CountVectorizer(min_df=2, max_df=0.7, stop_words=stopwords.words('english'))
t1 = countVectorizer.fit_transform(df['Sentence'].values.astype('U')).toarray()
vocabulary = countVectorizer.get_feature_names_out()
vectorizer = CountVectorizer(min_df=2, max_df=0.7, stop_words=stopwords.words('english'), vocabulary=vocabulary)

X_train = vectorizer.fit_transform(train['Sentence']).toarray()
y_train = train['Label'].values

X_test = vectorizer.fit_transform(test['Sentence']).toarray()
y_test = test['Label'].values
# (sentence_index, feature_index) count
print(X_train)

# Linear Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

model = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred = model.predict(X_test)
a = accuracy_score(y_test, y_pred)
precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

print("Accuracy : {:.4f}".format(a))
print("Precision : {:.4f}".format(precision))
print("Recall : {:.4f}".format(recall))
print("F-Score : {:.4f}".format(fscore))
