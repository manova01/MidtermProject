import pickle
from flask import Flask, request, jsonify

model_file = 'model_C=10.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('not_reached')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    not_reached = y_pred >= 0.5

    result = {
        'Not_reached_probability': float(y_pred),
        'not_reached': bool(not_reached)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
