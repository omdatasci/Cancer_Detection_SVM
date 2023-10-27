import pandas as pd
import numpy as np
import pickle

from flask import Flask, render_template, request
app = Flask(__name__)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'Clump' : float(request.form['clump']),
        'UnifSize' : float(request.form['unifSize']),
        'UnifShape' : float(request.form['unifShape']),
        'MargAdh' : float(request.form['margAdh']),
        'SingEpiSize' : float(request.form['singEpiSize']),
        'BlandChrom' : float(request.form['blandChrom']),
        'NormNucl' : float(request.form['normNucl']),
    }

    data_df = pd.DataFrame([data])
    prediction = model.predict(data_df)[0]

    return render_template('result.html', predictions = prediction)

if __name__ == '__main__':
    app.run(debug=True)