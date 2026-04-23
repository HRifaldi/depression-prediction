# Judul Project

## Repository Outline
`Bagian ini menjelaskan secara singkat konten/isi dari file yang dipush ke repository`

Contoh:
```
1. README.md - Penjelasan gambaran umum project
2. notebook.ipynb - Notebook yang berisi pengolahan data dengan python
dst.
```

# Problem Background

`Kesehatan mental pada mahasiswa merupakan isu penting karena tekanan akademik, jam belajar/kerja, kondisi finansial, dan kebiasaan hidup dapat berhubungan dengan meningkatnya risiko depresi. Project ini dibuat untuk membantu memahami pola data terkait depresi pada mahasiswa melalui analisis sederhana (EDA) dan membangun model klasifikasi yang dapat memprediksi kemungkinan depresi berdasarkan karakteristik individu.`

# Project Output

`Output dari project ini adalah:`
`1. Model machine learning klasifikasi untuk memprediksi status depression (0/1).`
`2. Aplikasi Streamlit yang berisi:`
    `Halaman EDA (visualisasi sederhana)`
    `Halaman Prediction`

# Data

`Dataset yang digunakan adalah data “student depression” (berisi kombinasi fitur numerik dan kategorikal) dengan target depression.`
`Karakteristik data secara umum:`
    `Memiliki fitur numerik seperti age, cgpa, academic_pressure, financial_stress, work_study_hours dan lainnya.`
    `Memiliki fitur kategorikal seperti gender, sleep_duration, dietary_habits, suicidal_thoughts, family_history_mental_illness, profession, degree, dll.`

`Pada preprocessing, nilai kosong ditangani dengan imputasi:`
`Numerik: median`
`Kategorikal: most_frequent`
`Ada nilai tidak valid seperti tanda ? pada kolom Financial Stress yang dibersihkan sebelum training.`

# Method

`Project ini menggunakan pendekatan Supervised Learning (Classification), dengan tahapan utama:`
1. Preprocessing
2. Pisahkan fitur numerik dan kategorikal.
3. Numerik: imputasi median + standardisasi (StandardScaler).
4. Kategorikal: imputasi most_frequent + one-hot encoding (OneHotEncoder).
5. Semua preprocessing digabung dalam ColumnTransformer dan dimasukkan ke Pipeline.

# Modeling & Selection

`Mencoba beberapa model klasifikasi: KNN, SVM (LinearSVC), Decision Tree, Random Forest, AdaBoost. Memilih model terbaik berdasarkan evaluasi (CV/tuning jika ada).`

`Evaluation`
`Menampilkan metrik evaluasi seperti classification report, confusion matrix, dan ROC curve untuk mengukur performa model di data test.`

# Stacks

Language: Python
Tools: Jupyter Notebook, Streamlit, Docker
Libraries:
pandas, numpy
scikit-learn (Pipeline, ColumnTransformer, model klasifikasi, evaluasi)
matplotlib (visualisasi)
joblib (menyimpan & load model)
streamlit (deployment dashboard)

# Reference

`Notebook training & evaluasi: PIM2_Hernanda_Rifaldi_Depression.ipynb, https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset/code`
`Aplikasi Streamlit: deployment/src/streamlit_app.py`
`Model pipeline: artifacts/best_model.joblib`