from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Buat instance Flask
app = Flask(__name__)

# Muat model yang telah disimpan
model = joblib.load('linear_regression_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari request
        data = request.json
        # Misalkan data dikirim dalam bentuk JSON: {"input": [1.5, 2.5, 3.5]}
        input_data = np.array(data['input']).reshape(-1, 1)
        
        # Buat prediksi menggunakan model
        predictions = model.predict(input_data)
        
        # Kembalikan hasil prediksi sebagai JSON
        return jsonify({'predictions': predictions.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
