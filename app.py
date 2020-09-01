from flask import Flask, request, jsonify, render_template, session, url_for, redirect
import numpy as np
from wtforms import TextField, SubmitField
import tensorflow
from keras.models import load_model
import joblib

from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route("/home", methods=['GET', 'POST'])
def home():
    return render_template('home.html')


model = load_model('ANN1.h5')
scaler = joblib.load("model1.pkl", 'r')


@app.route('/prediction', methods=['POST'])
def prediction():

    t_ = float(request.form['t'])
    TM_ = float(request.form['tM'])
    Tm_ = float(request.form['tm'])
    SLP_ = float(request.form['slp'])
    H_ = float(request.form['h'])
    VV_ = float(request.form['vv'])
    V_ = float(request.form['v'])
    VM_ = float(request.form['vm'])
    content = [[t_, TM_, Tm_, SLP_, H_, VV_, V_, VM_]]

    """content['T'] = float(request.form['t'])
    content['TM'] = float(request.form['tM'])
    content['Tm'] = float(request.form['tm'])
    content['SLP'] = float(request.form['slp'])
    content['H'] = float(request.form['h'])
    content['VV'] = float(request.form['vv'])
    content['V'] = float(request.form['v'])
    content['VM'] = float(request.form['vm'])"""
    content = scaler.transform(content)
    result = model.predict(content)[0][0]
    return render_template('prediction.html', results=result)


if __name__ == '__main__':
    app.run(debug=True)
