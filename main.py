from flask import Flask, flash, redirect, url_for, request, render_template
import tensorflow as tf
import pickle
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

@app.route('/')
def index():
    return render_template('index.html')
 
@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        passwd = request.form['pass']
        corpus = [passwd]
        vectorizer = pickle.load(open("vectorizer_cnn.obj", 'rb'))
        count_matrix = vectorizer.transform(corpus).toarray()
        print(count_matrix.shape)
        loaded_model = tf.keras.saving.load_model("model.keras")
        
        y_pred = loaded_model.predict(count_matrix)
        if y_pred > 0.5:
            flash('We found SQLi', 'danger')
            return redirect(url_for('index'))
        if y_pred <= 0.5:
            flash('It seems to be safe', 'success')
            return redirect(url_for('index'))
    else:
        return render_template('index.html')
 
if __name__ == '__main__':
    app.run()