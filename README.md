# Student Depression Risk Analysis and Prediction

> A classification project to analyze student mental-health risk factors and predict `depression` status using a Streamlit app.

* * *

## Project Overview

This project focuses on identifying depression-related patterns in student data and building a deployable machine learning model for practical inference.

Main goals:
1. Explore how depression is distributed in the dataset.
2. Train and tune multiple classification models.
3. Select the best tuned model based on cross-validation performance.
4. Evaluate the tuned model on a hold-out test set.
5. Serve EDA and prediction via Streamlit.

* * *

## Repository Structure

```text
|-- README.md
|-- description.md
|-- PIM2_Hernanda_Rifaldi_Depression.ipynb
|-- PIM2_Hernanda_Rifaldi_inf.ipynb
|-- student_depression_dataset.csv
|-- artifacts/
|   `-- best_model.joblib
`-- deployment/
    |-- requirements.txt
    |-- Dockerfile
    `-- src/
        |-- streamlit_app.py
        |-- eda.py
        |-- prediction.py
        |-- best_model.joblib
        `-- student_depression_dataset.csv
```

* * *

## Dataset

- Source: [Kaggle - Student Depression Dataset](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset)
- Target column: `depression` (0 = No Depression, 1 = Depression)
- Feature types: mixed numerical and categorical

Example numerical features:
- `age`
- `cgpa`
- `academic_pressure`
- `work_study_hours`
- `financial_stress`

Example categorical features:
- `gender`
- `city`
- `profession`
- `sleep_duration`
- `dietary_habits`
- `degree`
- `suicidal_thoughts`
- `family_history_mental_illness`

* * *

## Modeling Pipeline

Preprocessing (`ColumnTransformer`):
- Numerical: `SimpleImputer(strategy="median")` + `StandardScaler`
- Categorical: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`

Models tuned:
- SVM (LinearSVC)
- Random Forest
- AdaBoost
- KNN

Hyperparameter search:
- `RandomizedSearchCV`
- 5-fold `StratifiedKFold`
- Optimization metric: `F1`

* * *

## Tuned Model Results Only

### Tuning leaderboard (cross-validation, after tuning)

| Model | Best CV F1 | CV Std | Best Parameters |
|---|---:|---:|---|
| AdaBoost | 0.8737 | 0.0021 | `learning_rate=0.3012291401980419`, `n_estimators=400` |
| SVM | 0.8730 | 0.0020 | `C=0.008111941985431923` |
| KNN | 0.8694 | 0.0016 | `weights='distance'`, `n_neighbors=27` |
| RandomForest | 0.8684 | 0.0019 | `n_estimators=400`, `min_samples_split=5`, `min_samples_leaf=1`, `max_depth=None` |

Selected final model: **AdaBoost (tuned)**.

### Hold-out test metrics (tuned AdaBoost only)

| Metric | Value |
|---|---:|
| Accuracy | 0.8464 |
| ROC-AUC | 0.9203 |

Class-wise metrics:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 (No Depression) | 0.8318 | 0.7890 | 0.8099 | 2313 |
| 1 (Depression) | 0.8559 | 0.8871 | 0.8712 | 3268 |
| Macro Avg | 0.8439 | 0.8381 | 0.8405 | 5581 |
| Weighted Avg | 0.8459 | 0.8464 | 0.8458 | 5581 |

Confusion matrix (tuned AdaBoost):

| Actual \\ Predicted | 0 | 1 |
|---|---:|---:|
| 0 | 1825 | 488 |
| 1 | 369 | 2899 |

* * *

## Streamlit App

The deployment app includes:
- **EDA page** for data exploration and visual summaries.
- **Prediction page** for manual input and depression risk prediction.

Run locally:

```bash
pip install -r deployment/requirements.txt
cd deployment/src
streamlit run streamlit_app.py
```

* * *

## Author

Hernanda Rifaldi

## License

This repository is for learning and portfolio purposes.
