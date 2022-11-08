from flask import Flask
from flask import request
from flask import jsonify

import pickle

def load(filename):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)


dv = load('dv.bin')
model = load('model.bin')

app = Flask('arrest')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    X = dv.transform([data])
    y_pred = model.predict_proba(X)[0, 1]
    arrest = y_pred >= 0.5

    result = {
        'arrest_probability': float(y_pred),
        'arrest': bool(arrest)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)