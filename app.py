import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import os

# 1. Inisialisasi Flask
flask_app = Flask(__name__)

# 2. Muat Model dan Transformer
# Gunakan nama variabel yang sangat spesifik
model_obj = None
transformer_obj = None

try:
    with open("linear_regression_model.pkl", "rb") as f1:
        model_obj = pickle.load(f1)
    
    with open("transformer.pkl", "rb") as f2:
        transformer_obj = pickle.load(f2)
    
    print("Model & Transformer successfully loaded!")
except Exception as e:
    print(f"CRITICAL ERROR LOADING PKL: {e}")

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    try:
        # A. Ambil data dari form
        age = request.form.get('Age')
        gender = request.form.get('Gender')
        blood_type = request.form.get('Blood_Type')
        condition = request.form.get('Medical_Condition')

        # B. Cek apakah objek model sudah termuat
        if model_obj is None or transformer_obj is None:
            return render_template("index.html", prediction_text="Error: Model files not loaded on server.")

        # C. Susun ke dalam DataFrame
        # Pastikan nama kolom sama persis dengan 'regression.py'
        df_input = pd.DataFrame([[int(age), gender, blood_type, condition]], 
                                columns=['Age', 'Gender', 'Blood Type', 'Medical Condition'])
        
        # D. Transformasi Data
        # Kita panggil variabel transformer_obj (pastikan ini bukan string)
        transformed_data = transformer_obj.transform(df_input)
        
        # E. Prediksi
        prediction = model_obj.predict(transformed_data)
        
        # F. Format hasil
        result = "{:,.2f}".format(prediction[0])
        
        return render_template("index.html", 
                               prediction_text=f"Estimated Billing Amount: ${result}")

    except Exception as e:
        return render_template("index.html", 
                               prediction_text=f"Error Detail: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    flask_app.run(host='0.0.0.0', port=port)











# import numpy as np
# import pandas as pd
# from flask import Flask, request, render_template
# import pickle
# import os

# # 1. Inisialisasi Flask
# flask_app = Flask(__name__)

# # 2. Muat Model dan Transformer
# # Gunakan try-except untuk memantau log jika file tidak ditemukan
# try:
#     with open("linear_regression_model.pkl", "rb") as model_file:
#         model_billing = pickle.load(model_file)
    
#     with open("transformer.pkl", "rb") as transformer_file:
#         transformer_billing = pickle.load(transformer_file)
# except Exception as e:
#     print(f"Gagal memuat file pkl: {e}")

# @flask_app.route("/")
# def Home():
#     return render_template("index.html")

# @flask_app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # A. Ambil data dari form HTML
#         age = request.form.get('Age')
#         gender = request.form.get('Gender')
#         blood_type = request.form.get('Blood_Type')
#         condition = request.form.get('Medical_Condition')

#         # B. Susun ke dalam DataFrame
#         # Pastikan nama kolom SAMA PERSIS dengan saat training (cek spasi di 'Blood Type')
#         df_input = pd.DataFrame([[int(age), gender, blood_type, condition]], 
#                                 columns=['Age', 'Gender', 'Blood Type', 'Medical Condition'])
        
#         # C. Transformasi data
#         # Kadang transformer butuh DataFrame, kadang butuh array. 
#         # Jika .values tadi masih error, kita gunakan DataFrame tapi pastikan versinya sinkron.
#         # Untuk kasusmu, coba gunakan df_input langsung atau df_input.values
#         transformed_data = transformer_billing.transform(df_input)
        
#         # D. Jalankan Prediksi
#         prediction = model_billing.predict(transformed_data)
        
#         # E. Format hasil angka
#         # Menggunakan format {:,} agar ada pemisah ribuan (contoh: 35,000.00)
#         result = "{:,.2f}".format(prediction[0])
        
#         return render_template("index.html", 
#                                prediction_text=f"Estimated Billing Amount: ${result}")

#     except Exception as e:
#         # Menampilkan detail error di halaman jika terjadi kegagalan
#         return render_template("index.html", 
#                                prediction_text=f"Error Detail: {str(e)}")

# if __name__ == "__main__":
#     # Render membutuhkan port dinamis
#     port = int(os.environ.get("PORT", 5000))
#     flask_app.run(host='0.0.0.0', port=port)


# # import numpy as np
# # import pandas as pd
# # from flask import Flask, request, render_template
# # import pickle
# # import os

# # # 1. Inisialisasi Flask
# # flask_app = Flask(__name__)

# # # 2. Muat Model dan Transformer
# # # Menggunakan nama variabel unik untuk menghindari tabrakan dengan tipe data lain
# # try:
# #     with open("linear_regression_model.pkl", "rb") as model_file:
# #         model_billing = pickle.load(model_file)
    
# #     with open("transformer.pkl", "rb") as transformer_file:
# #         transformer_billing = pickle.load(transformer_file)
# # except Exception as e:
# #     print(f"Gagal memuat file pkl: {e}")

# # @flask_app.route("/")
# # def Home():
# #     return render_template("index.html")

# # @flask_app.route("/predict", methods=["POST"])
# # def predict():
# #     try:
# #         # A. Ambil data dari form HTML
# #         age = request.form.get('Age')
# #         gender = request.form.get('Gender')
# #         blood_type = request.form.get('Blood_Type')
# #         condition = request.form.get('Medical_Condition')

# #         # B. Susun ke dalam DataFrame dengan URUTAN yang benar
# #         # Urutan wajib: Age -> Gender -> Blood Type -> Medical Condition
# #         df_input = pd.DataFrame([[int(age), gender, blood_type, condition]], 
# #                                 columns=['Age', 'Gender', 'Blood Type', 'Medical Condition'])
        
# #         # C. SOLUSI UTAMA: Gunakan .values
# #         # Ini akan mengirim data dalam bentuk array murni ke transformer
# #         # sehingga Sklearn tidak akan komplain soal "Feature Names Mismatch"
# #         input_data = df_input.values
        
# #         # D. Jalankan Transformasi (OneHotEncoding)
# #         transformed_data = transformer_billing.transform(input_data)
        
# #         # E. Jalankan Prediksi menggunakan model Linear Regression
# #         prediction = model_billing.predict(transformed_data)
        
# #         # F. Format hasil angka (ambil index 0 karena outputnya array)
# #         result = "{:,.2f}".format(prediction[0])
        
# #         return render_template("index.html", 
# #                                prediction_text=f"Estimated Billing Amount: ${result}")

# #     except Exception as e:
# #         # Menampilkan detail error di halaman jika terjadi kegagalan logika
# #         return render_template("index.html", 
# #                                prediction_text=f"Error Detail: {str(e)}")

# # if __name__ == "__main__":
# #     # Penting: Gunakan os.environ agar Render bisa menentukan Port secara dinamis
# #     port = int(os.environ.get("PORT", 5000))
# #     flask_app.run(host='0.0.0.0', port=port)

# # import numpy as np
# # import pandas as pd
# # from flask import Flask, request, render_template
# # import pickle
# # import os

# # flask_app = Flask(__name__)

# # # Load model dan transformer
# # try:
# #     model_billing = pickle.load(open("linear_regression_model.pkl", "rb"))
# #     transformer_billing = pickle.load(open("transformer.pkl", "rb"))
# # except Exception as e:
# #     print(f"Error loading files: {e}")

# # @flask_app.route("/")
# # def Home():
# #     return render_template("index.html")

# # @flask_app.route("/predict", methods=["POST"])
# # def predict():
# #     try:
# #         # 1. Ambil data dari form
# #         age = request.form.get('Age')
# #         gender = request.form.get('Gender')
# #         blood_type = request.form.get('Blood_Type')
# #         condition = request.form.get('Medical_Condition')

# #         # 2. Susun ke dalam DataFrame
# #         # Urutan harus PERSIS: Age, Gender, Blood Type, Medical Condition
# #         df_input = pd.DataFrame([[int(age), gender, blood_type, condition]], 
# #                                 columns=['Age', 'Gender', 'Blood Type', 'Medical Condition'])
        
# #         # 3. Transformasi data
# #         # TIPS: Kita gunakan .values untuk menghindari error "Feature names mismatch"
# #         # Ini memaksa sklearn hanya melihat urutan kolom, bukan namanya.
# #         input_data = df_input.values
# #         transformed_data = transformer_billing.transform(input_data)
        
# #         # 4. Prediksi
# #         prediction = model_billing.predict(transformed_data)
# #         output = "{:,.2f}".format(prediction[0])
        
# #         return render_template("index.html", 
# #                                prediction_text=f"Estimated Billing Amount: ${output}")

# #     except Exception as e:
# #         # Jika error, tampilkan detailnya di halaman web
# #         return render_template("index.html", prediction_text=f"Error Detail: {str(e)}")

# # if __name__ == "__main__":
# #     port = int(os.environ.get("PORT", 5000))
# #     flask_app.run(host='0.0.0.0', port=port)

# # import numpy as np
# # import pandas as pd
# # from flask import Flask, request, render_template
# # import pickle
# # import os

# # # 1. Inisialisasi Flask app
# # flask_app = Flask(__name__)

# # # 2. Muat Model dan Transformer secara global (di luar fungsi)
# # # Kita gunakan try-except agar jika file pkl hilang, aplikasi memberikan info di log
# # try:
# #     with open("linear_regression_model.pkl", "rb") as model_file:
# #         model_billing = pickle.load(model_file)
    
# #     with open("transformer.pkl", "rb") as transformer_file:
# #         transformer_billing = pickle.load(transformer_file)
# # except Exception as e:
# #     print(f"Gagal memuat file pkl: {e}")

# # @flask_app.route("/")
# # def Home():
# #     return render_template("index.html")

# # @flask_app.route("/predict", methods=["POST"])
# # def predict():
# #     try:
# #         # A. Ambil data dari form secara spesifik berdasarkan atribut 'name' di HTML
# #         age = request.form.get('Age')
# #         gender = request.form.get('Gender')
# #         blood_type = request.form.get('Blood_Type')
# #         medical_condition = request.form.get('Medical_Condition')

# #         # B. Validasi: Pastikan tidak ada input yang kosong
# #         if not all([age, gender, blood_type, medical_condition]):
# #             return render_template("index.html", prediction_text="Error: Harap isi semua kolom!")

# #         # C. Buat DataFrame dengan nama kolom yang PERSIS sama dengan saat training (regression.py)
# #         # Penting: Age harus diubah ke int agar Linear Regression bisa menghitungnya
# #         df_input = pd.DataFrame([[int(age), gender, blood_type, medical_condition]], 
# #                                 columns=['Age', 'Gender', 'Blood Type', 'Medical Condition'])
        
# #         # D. Transformasi data kategorikal (Gender, dsb) menjadi matrix angka 0/1 (One Hot)
# #         # Kita gunakan variabel transformer_billing yang sudah di-load di atas
# #         transformed_data = transformer_billing.transform(df_input)
        
# #         # E. Lakukan Prediksi
# #         prediction = model_billing.predict(transformed_data)
        
# #         # F. Ambil hasil angka pertama dan format ke mata uang
# #         output = "{:,.2f}".format(prediction[0])
        
# #         return render_template("index.html", 
# #                                prediction_text=f"Estimated Billing Amount: ${output}")

# #     except Exception as e:
# #         # Jika terjadi error logika (seperti mismatch kolom), pesan error tampil di web
# #         return render_template("index.html", prediction_text=f"Error: {str(e)}")

# # if __name__ == "__main__":
# #     # Gunakan port dari environment variable (penting untuk Render)
# #     port = int(os.environ.get("PORT", 5000))
# #     flask_app.run(host='0.0.0.0', port=port, debug=True)

# # import numpy as np
# # import pandas as pd
# # from flask import Flask, request, render_template
# # import pickle

# # # Inisialisasi Flask
# # flask_app = Flask(__name__)

# # # Memuat model dan transformer
# # try:
# #     model = pickle.load(open("linear_regression_model.pkl", "rb"))
# #     transformer = pickle.load(open("transformer.pkl", "rb"))
# # except Exception as e:
# #     print(f"Error loading models: {e}")

# # @flask_app.route("/")
# # def Home():
# #     return render_template("index.html")

# # @flask_app.route("/predict", methods=["POST"])
# # def predict():
# #     try:
# #         # 1. Ambil data spesifik dari Form
# #         age = request.form.get('Age')
# #         gender = request.form.get('Gender')
# #         blood_type = request.form.get('Blood_Type')
# #         condition = request.form.get('Medical_Condition')

# #         # 2. Buat DataFrame (Data harus TEKS asli agar bisa diproses OneHotEncoder)
# #         # Pastikan nama kolom SAMA PERSIS dengan saat training di regression.py
# #         df_input = pd.DataFrame([[int(age), gender, blood_type, condition]], 
# #                                 columns=['Age', 'Gender', 'Blood Type', 'Medical Condition'])
        
# #         # 3. Transformasi data menggunakan transformer (Jembatan antara teks ke angka One-Hot)
# #         transformed_data = transformer.transform(df_input)
        
# #         # 4. Prediksi menggunakan model
# #         prediction = model.predict(transformed_data)
        
# #         # 5. Format hasil ke dua angka di belakang koma
# #         output = "{:,.2f}".format(prediction[0])
        
# #         return render_template("index.html", 
# #                                prediction_text=f"Estimated Billing Amount: ${output}")

# #     except Exception as e:
# #         # Menampilkan error spesifik di halaman jika terjadi kegagalan logika
# #         return render_template("index.html", prediction_text=f"Error: {str(e)}")

# # if __name__ == "__main__":
# #     flask_app.run(debug=True)


# # import numpy as np
# # import pandas as pd # Tambahkan pandas
# # from flask import Flask, request, render_template
# # import pickle

# # # Create flask app
# # flask_app = Flask(__name__)

# # # Load model DAN transformer
# # model = pickle.load(open("linear_regression_model.pkl", "rb"))
# # transformer = pickle.load(open("transformer.pkl", "rb"))

# # @flask_app.route("/")
# # def Home():
# #     return render_template("index.html")

# # @flask_app.route("/predict", methods=["POST"])
# # def predict():
# #     # 1. Ambil data dari form
# #     data = [x for x in request.form.values()]
    
# #     # 2. Bungkus ke DataFrame agar sesuai dengan urutan kolom saat training
# #     # Urutan: Age, Gender, Blood Type, Medical Condition
# #     df_input = pd.DataFrame([data], columns=['Age', 'Gender', 'Blood Type', 'Medical Condition'])
    
# #     # 3. Transform data menggunakan transformer yang sudah di-load
# #     transformed_data = transformer.transform(df_input)
    
# #     # 4. Prediksi menggunakan model
# #     prediction = model.predict(transformed_data)
    
# #     # 5. Format hasil (Ambil angka pertama dan bulatkan)
# #     output = round(prediction[0], 2)
    
# #     return render_template("index.html", 
# #                            prediction_text="Estimated Billing Amount: ${}".format(output))

# # if __name__ == "__main__":
# #     flask_app.run(debug=True)


# # import numpy as np
# # from flask import Flask, request, jsonify, render_template
# # import pickle

# # # Create flask app
# # flask_app = Flask(__name__)
# # model = pickle.load(open("linear_regression_model.pkl", "rb"))

# # @flask_app.route("/")
# # def Home():
# #     return render_template("index.html")

# # @flask_app.route("/predict", methods = ["POST"])
# # def predict():
# #     float_features = [float(x) for x in request.form.values()]
# #     features = [np.array(float_features)]
# #     prediction = model.predict(features)
# #     return render_template("index.html", prediction_text = "The flower species is {}".format(prediction))

# # if __name__ == "__main__":
# #     flask_app.run(debug=True)
