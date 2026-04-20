import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

# Inisialisasi Flask
flask_app = Flask(__name__)

# Memuat model dan transformer
try:
    model = pickle.load(open("linear_regression_model.pkl", "rb"))
    transformer = pickle.load(open("transformer.pkl", "rb"))
except Exception as e:
    print(f"Error loading models: {e}")

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Ambil data spesifik dari Form
        age = request.form.get('Age')
        gender = request.form.get('Gender')
        blood_type = request.form.get('Blood_Type')
        condition = request.form.get('Medical_Condition')

        # 2. Buat DataFrame (Data harus TEKS asli agar bisa diproses OneHotEncoder)
        # Pastikan nama kolom SAMA PERSIS dengan saat training di regression.py
        df_input = pd.DataFrame([[int(age), gender, blood_type, condition]], 
                                columns=['Age', 'Gender', 'Blood Type', 'Medical Condition'])
        
        # 3. Transformasi data menggunakan transformer (Jembatan antara teks ke angka One-Hot)
        transformed_data = transformer.transform(df_input)
        
        # 4. Prediksi menggunakan model
        prediction = model.predict(transformed_data)
        
        # 5. Format hasil ke dua angka di belakang koma
        output = "{:,.2f}".format(prediction[0])
        
        return render_template("index.html", 
                               prediction_text=f"Estimated Billing Amount: ${output}")

    except Exception as e:
        # Menampilkan error spesifik di halaman jika terjadi kegagalan logika
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    flask_app.run(debug=True)


# import numpy as np
# import pandas as pd # Tambahkan pandas
# from flask import Flask, request, render_template
# import pickle

# # Create flask app
# flask_app = Flask(__name__)

# # Load model DAN transformer
# model = pickle.load(open("linear_regression_model.pkl", "rb"))
# transformer = pickle.load(open("transformer.pkl", "rb"))

# @flask_app.route("/")
# def Home():
#     return render_template("index.html")

# @flask_app.route("/predict", methods=["POST"])
# def predict():
#     # 1. Ambil data dari form
#     data = [x for x in request.form.values()]
    
#     # 2. Bungkus ke DataFrame agar sesuai dengan urutan kolom saat training
#     # Urutan: Age, Gender, Blood Type, Medical Condition
#     df_input = pd.DataFrame([data], columns=['Age', 'Gender', 'Blood Type', 'Medical Condition'])
    
#     # 3. Transform data menggunakan transformer yang sudah di-load
#     transformed_data = transformer.transform(df_input)
    
#     # 4. Prediksi menggunakan model
#     prediction = model.predict(transformed_data)
    
#     # 5. Format hasil (Ambil angka pertama dan bulatkan)
#     output = round(prediction[0], 2)
    
#     return render_template("index.html", 
#                            prediction_text="Estimated Billing Amount: ${}".format(output))

# if __name__ == "__main__":
#     flask_app.run(debug=True)


# import numpy as np
# from flask import Flask, request, jsonify, render_template
# import pickle

# # Create flask app
# flask_app = Flask(__name__)
# model = pickle.load(open("linear_regression_model.pkl", "rb"))

# @flask_app.route("/")
# def Home():
#     return render_template("index.html")

# @flask_app.route("/predict", methods = ["POST"])
# def predict():
#     float_features = [float(x) for x in request.form.values()]
#     features = [np.array(float_features)]
#     prediction = model.predict(features)
#     return render_template("index.html", prediction_text = "The flower species is {}".format(prediction))

# if __name__ == "__main__":
#     flask_app.run(debug=True)
