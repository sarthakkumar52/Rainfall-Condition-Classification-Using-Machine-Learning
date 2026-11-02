# 🌧️ Rainfall Condition Classification Using Machine Learning

## 📌 Project Overview

This project uses Machine Learning to classify rainfall conditions into five categories:

- Extreme Dry
- Dry
- Normal
- Wet
- Extreme Wet

The project uses historical rainfall data from **36 IMD subdivisions of India**, covering the period **1901–2017**.

The main objective is to use **previous rainfall patterns and historical information** to classify the rainfall condition of the following year.

---

## 🎯 Problem Statement

Rainfall patterns vary significantly across different regions and years. This project develops a Machine Learning classification system that identifies the expected rainfall condition based on historical rainfall patterns.

This is a **classification problem**, not an exact rainfall amount prediction problem.

---

## 📊 Dataset

The dataset contains rainfall information for 36 Indian meteorological subdivisions.

**Period:** 1901–2017

Important features include:

- Monthly rainfall: JAN–DEC
- Annual rainfall
- Seasonal rainfall:
  - JF
  - MAM
  - JJAS
  - OND
- IMD Subdivision
- Year

### Target Classes

Rainfall is classified into five categories:

| Class | Meaning |
|---|---|
| Extreme Dry | Very low rainfall |
| Dry | Below-normal rainfall |
| Normal | Normal rainfall |
| Wet | Above-normal rainfall |
| Extreme Wet | Very high rainfall |

---

## ⚙️ Feature Engineering

To avoid data leakage, only information available from previous years is used for prediction.

Important engineered features include:

- Previous Annual Rainfall
- Previous Monthly Rainfall
- Previous Seasonal Rainfall
- 3-Year Rolling Mean
- 5-Year Rolling Mean
- Rainfall Change
- Rainfall Change Percentage
- IMD Subdivision

---

## 🤖 Machine Learning Algorithms

Five classification algorithms were implemented:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Naive Bayes
4. Decision Tree
5. Support Vector Machine (SVM)

Hyperparameter tuning was performed to improve model performance.

---

## 📈 Model Evaluation

Because the dataset contains imbalanced rainfall classes, **Macro F1 Score** was used as the primary metric for model selection.

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| KNN | 0.5494 | **0.2256** |
| SVM | 0.3210 | 0.2184 |
| Decision Tree | 0.2988 | 0.1847 |
| Logistic Regression | 0.1926 | 0.1668 |
| Naive Bayes | 0.1247 | 0.0965 |

### 🏆 Best Model

**KNN (K-Nearest Neighbors)** was selected as the final model because it achieved the highest Macro F1 Score of **0.2256**.

The best K value was **3**, using distance-based weighting.

---

## 🔍 Sample Prediction

The final KNN model was tested on a sample from the test dataset.

**Actual Class:** Normal

**Predicted Class:** Normal

The sample prediction was correct.

---

## 📌 Key Findings

- Normal rainfall is the most frequent class.
- Rainfall patterns vary considerably across IMD subdivisions.
- JJAS (monsoon season) contributes the largest amount of average seasonal rainfall.
- KNN achieved the best overall Macro F1 score.
- Minority classes, especially Extreme Wet, were difficult for the models to classify accurately.

---

## ⚠️ Limitations

The current model uses mainly historical rainfall information. It does not include additional meteorological variables such as:

- Temperature
- Humidity
- Wind speed
- Atmospheric pressure
- Satellite observations

The class imbalance also affects minority-class prediction performance.

---

## 🚀 Future Scope

Future improvements could include:

- More recent rainfall data
- Additional meteorological variables
- Satellite and remote-sensing data
- Climate indicators
- XGBoost / LightGBM
- Class-balancing techniques
- Deep Learning and time-series models
- Spatial modelling

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook / Spyder

---

## 📚 Dataset Source

The dataset is based on the **Sub Divisional Monthly Rainfall from 1901 to 2017** dataset provided through the Government of India's Open Government Data platform and India Meteorological Department (IMD).

---

## 👨‍💻 Project

**Rainfall Condition Classification Using Machine Learning**

Machine Learning Classification Project
