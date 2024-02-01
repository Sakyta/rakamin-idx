from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
import numpy as np
import pickle
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'xgb.pkl')
scaler_path = os.path.join(script_dir, 'scaler.pkl')

model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['post'])
def predict():
    recoveries = float(request.form['recoveries'])
    pymnt_time = float(request.form['pymnt_time'])
    out_prncp = float(request.form['out_prncp'])
    term = float(request.form['term'])
    late_fee = float(request.form['late_fee'])
    int_rate = float(request.form['int_rate'])
    initial_list_status = float(request.form['initial_list_status'])
    arr = np.array([[recoveries, pymnt_time, out_prncp, late_fee, term, int_rate, initial_list_status]])
    scaled = scaler.transform(arr)
    pred = model.predict(scaled)
    prob = model.predict_proba(scaled)
    
    prediction = 'Good Loan' if pred[0] == 1 else 'Bad Loan'
    probability = str(prob[0][0])
    
    return jsonify({'prediction': prediction, 'probability': probability})

if __name__ == "__main__":
    app.run(debug=True)