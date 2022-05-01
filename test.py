from flask import Flask, render_template, request, url_for, jsonify
import numpy as np
import pickle
import nltk
import re

app = Flask(__name__)

multi_model = pickle.load( open("multi_pickle.p", "rb"))
with open('count_vec.p', 'rb') as f:
    bow_transformer = pickle.load(f)


@app.route('/index.html')
def hello_world3():
    return render_template("index.html")

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/analysis.html')
def hello_world2():
    return render_template("analysis.html")

@app.route('/analysis.html', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        message = request.form['message']
        sentences = re.split("(?<=[.!?]) +", message)
        sent = bow_transformer.transform(sentences).toarray()
        result = multi_model.predict(sent)
        result = result[0]
        return render_template('analysis.html', predictions=result, mes = message)

if __name__ =="__main__":
    app.run(debug=True) 