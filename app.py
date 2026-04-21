import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import os

flask_app = Flask(__name__)

# Load Model & Transformer secara langsung agar server error jika gagal (fail-fast)
try:
    model_obj = pickle.load(open("linear_regression_model.pkl", "rb"))
    transformer_obj = pickle.load(open("transformer.pkl", "rb"))
    print("SUCCESS: Files loaded!")
except Exception as e:
    # Ini akan muncul di logs Render jika file pkl bermasalah
    model_obj = None
    transformer_obj = None
    print(f"FAILED TO LOAD PKL: {e}")

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    try:
        # Cek ketersediaan objek
        if model_obj is None or transformer_obj is None:
            return render_template("index.html", 
                                   prediction_text="Error: Model/Transformer file not found on server or corrupted.")

        # Ambil data
        age = request.form.get('Age')
        gender = request.form.get('Gender')
        blood_type = request.form.get('Blood_Type')
        condition = request.form.get('Medical_Condition')

        # Buat DataFrame
        df_input = pd.DataFrame([[int(age), gender, blood_type, condition]], 
                                columns=['Age', 'Gender', 'Blood Type', 'Medical Condition'])
        
        # Gunakan .values untuk keamanan transformasi
        transformed_data = transformer_obj.transform(df_input.values)
        
        # Prediksi
        prediction = model_obj.predict(transformed_data)
        result = "{:,.2f}".format(prediction[0])
        
        return render_template("index.html", prediction_text=f"Estimated Billing: ${result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error Detail: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    flask_app.run(host='0.0.0.0', port=port)
