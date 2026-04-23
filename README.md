# Student Depression Risk Analysis and Prediction

> Proyek machine learning klasifikasi untuk menganalisis faktor depresi pada mahasiswa dan memprediksi status `depression` melalui aplikasi Streamlit.

* * *

## Background

Kesehatan mental mahasiswa menjadi isu penting karena dipengaruhi kombinasi tekanan akademik, kebiasaan hidup, jam belajar/kerja, serta kondisi finansial.  
Proyek ini dibuat untuk menjawab kebutuhan analisis data yang terstruktur sekaligus menyediakan prototipe prediksi yang bisa digunakan secara praktis.

* * *

## Objectives

### Pertanyaan utama

1. Bagaimana distribusi target `depression` pada dataset?
2. Fitur numerik apa yang paling membedakan kelompok depresi dan non-depresi?
3. Bagaimana pola depresi berdasarkan fitur kategorikal?
4. Model klasifikasi mana yang memberi performa terbaik?
5. Bagaimana model terbaik digunakan dalam aplikasi inference interaktif?

* * *

## Repository Structure

```text
|-- README.md                                 <- Dokumentasi utama proyek
|-- description.md                            <- Deskripsi project untuk kebutuhan milestone
|-- PIM2_Hernanda_Rifaldi_Depression.ipynb    <- Notebook utama (EDA, preprocessing, modeling, evaluasi)
|-- PIM2_Hernanda_Rifaldi_inf.ipynb           <- Notebook inference
|-- student_depression_dataset.csv            <- Dataset mentah
|-- artifacts/
|   `-- best_model.joblib                     <- Model terbaik hasil training
`-- deployment/
    |-- requirements.txt
    |-- Dockerfile
    `-- src/
        |-- streamlit_app.py                  <- Entry point Streamlit
        |-- eda.py                            <- Halaman EDA
        |-- prediction.py                     <- Halaman prediksi
        |-- best_model.joblib                 <- Model untuk deployment
        `-- student_depression_dataset.csv    <- Dataset untuk EDA app
```

* * *

## Dataset

- Sumber: [Kaggle - Student Depression Dataset](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset)
- Ukuran file: sekitar 2.9 MB (CSV)
- Tipe fitur: campuran numerik dan kategorikal
- Target: `depression` (0 = No Depression, 1 = Depression)

Contoh fitur numerik:
- `age`
- `cgpa`
- `academic_pressure`
- `study_satisfaction`
- `work_study_hours`
- `financial_stress`

Contoh fitur kategorikal:
- `gender`
- `city`
- `profession`
- `sleep_duration`
- `dietary_habits`
- `degree`
- `suicidal_thoughts`
- `family_history_mental_illness`

* * *

## Data Preparation and Method

Pipeline preprocessing menggunakan `ColumnTransformer`:
- Numerik: `SimpleImputer(strategy="median")` + `StandardScaler`
- Kategorikal: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`

Model yang dibandingkan:
- KNN
- SVM (LinearSVC)
- Decision Tree
- Random Forest
- AdaBoost

Strategi evaluasi:
- `StratifiedKFold` 5-fold cross-validation
- Metrics: Accuracy, F1-score, ROC-AUC
- Hyperparameter tuning: `RandomizedSearchCV` (untuk model terpilih)

* * *

## Modeling Results

### Cross-validation (ringkas)

| Model | Accuracy (mean) | F1 (mean) | ROC-AUC (mean) |
|---|---:|---:|---:|
| SVM | 0.8478 | 0.8724 | 0.9213 |
| AdaBoost | 0.8458 | 0.8705 | 0.9203 |
| RandomForest | 0.8427 | 0.8684 | 0.9139 |
| KNN | 0.8190 | 0.8499 | 0.8757 |
| DecisionTree | 0.7752 | 0.8081 | 0.7684 |

### Hasil tuning terbaik
- Best tuned model: **AdaBoost**
- Best CV F1: **0.8737**

### Evaluasi test set (best model: AdaBoost)
- Accuracy: **0.8464**
- Class `0` (No Depression): Precision **0.8318**, Recall **0.7890**, F1 **0.8099**
- Class `1` (Depression): Precision **0.8559**, Recall **0.8871**, F1 **0.8712**

* * *

## Streamlit App

Aplikasi deployment berisi:
- **EDA page**: ringkasan dataset, distribusi target, distribusi numerik per target, dan heatmap korelasi
- **Prediction page**: form input fitur untuk prediksi status depresi dan probabilitas kelas

Entry point aplikasi:
- `deployment/src/streamlit_app.py`

* * *

## Tech Stack

- Python
- Jupyter Notebook
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Joblib

* * *

## Getting Started

### 1. Clone repository

```bash
git clone https://github.com/<username>/p1-ftds-m2-HRifaldi.git
cd p1-ftds-m2-HRifaldi
```

### 2. Install dependencies

```bash
pip install -r deployment/requirements.txt
```

### 3. Run Streamlit app

```bash
cd deployment/src
streamlit run streamlit_app.py
```

### 4. Open notebook

```bash
jupyter notebook PIM2_Hernanda_Rifaldi_Depression.ipynb
```

* * *

## Author

Hernanda Rifaldi

* * *

## License

Proyek ini dibuat untuk tujuan pembelajaran dan portofolio data science.
"# depression-prediction" 
