#import Library yang dibutuhkan
import streamlit as st
import pickle
import os

# Load model yang sudah dibuat dari Jupyter Notebook
model_path = 'GBT_heartDisease_model.pkl'

if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        rf_model = loaded_model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
else:
    st.error("File 'GBT_heartDisease_model.pkl' tidak ditemukan di direktori")

# Tampilan aplikasi Streamlit
st.title("Prediksi Potensi Penyakit Jantung")
st.write("Aplikasi ini berguna untuk membantu mengenali potensi penyakit jantung pada manusia berusia 21 - 79 tahun")

Age = st.slider("Age", 21, 79)
Sex = st.selectbox("Gender", ["F", "M"])
ChestPainType = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
RestingBP = st.number_input("Resting Blood Pressure", 0, 200)
Cholesterol = st.number_input("Cholesterol", 0, 603)
FastingBS = st.selectbox("Fasting BS", ["0", "1"])
RestingECG = st.selectbox("Resting ECG", ["LVH", "Normal", "ST"])
MaxHR = st.number_input("Max Heart Rate", 60, 202)
ExcerciseAngina = st.radio("Exercise Angina", ["N", "Y"])
Old_peak = st.slider("Old Peak", -3.0, 7.0, step=0.1)
ST_Slope = st.selectbox("ST Slope", ["Down", "Flat", "Up"])

# Ubah opsi input menjadi One-Hot features
input_sex_F = 1 if Sex == "F" else 0
input_sex_M = 1 if Sex == "M" else 0

input_cpain_ASY = 1 if ChestPainType == "ASY" else 0
input_cpain_ATA = 1 if ChestPainType == "ATA" else 0
input_cpain_NAP = 1 if ChestPainType == "NAP" else 0
input_cpain_TA = 1 if ChestPainType == "TA" else 0

input_fastbs = 1 if FastingBS == "1" else 0

input_restecg_LVH = 1 if RestingECG == "LVH" else 0
input_restecg_Normal = 1 if RestingECG == "Normal" else 0
input_restecg_ST = 1 if RestingECG == "ST" else 0

input_anginaY = 1 if ExcerciseAngina == "Y" else 0
input_anginaN = 1 if ExcerciseAngina == "N" else 0

input_STslope_down = 1 if ST_Slope == "Down" else 0
input_STslope_flat = 1 if ST_Slope == "Flat" else 0
input_STslope_up = 1 if ST_Slope == "Up" else 0

# Buat input ke dalam numpy array
input_data = [[
    input_sex_F, input_sex_M, input_cpain_ASY, input_cpain_ATA,
    input_cpain_NAP, input_cpain_TA, input_restecg_LVH,
    input_restecg_Normal, input_restecg_ST, input_anginaN,
    input_anginaY, input_STslope_down, input_STslope_flat,
    input_STslope_up, Age, RestingBP, Cholesterol, input_fastbs,
    MaxHR, Old_peak
]]

# Tampilkan data yang akan diinputkan
st.write("Data pasien yang akan diinput ke model:")
st.write(input_data)

# Buat fungsi untuk prediksi
if st.button("Prediksi"):
    if 'rf_model' in locals():
        rf_model_prediction = rf_model.predict(input_data)
        outcome = {0: 'Tidak Berpotensi sakit jantung', 1: 'Berpotensi sakit jantung'}
        st.write(f"Orang tersebut diprediksi **{outcome[rf_model_prediction[0]]}**")
    else:
        st.error("Model belum dimuat.")
