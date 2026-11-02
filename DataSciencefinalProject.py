import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('/Users/sarthakkumar/Downloads/Sub_Division_IMD_2017.csv')

print("Dataset Loaded Successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

df.head()

df.tail()

print("Columns in Dataset:\n")

for i, column in enumerate(df.columns, 1):
    print(i, ":", column)
    
df.info()

df.describe()

duplicate_count = df.duplicated().sum()

print("Duplicate Rows:", duplicate_count)

print("Number of Subdivisions:",
      df["SUBDIVISION"].nunique())

print("\nSubdivisions:")
print(df["SUBDIVISION"].unique())

print("Minimum Year:", df["YEAR"].min())
print("Maximum Year:", df["YEAR"].max())

missing_summary = df.isnull().sum()

print("Missing Values in Each Column:")
print(missing_summary)

missing_summary = missing_summary[missing_summary > 0]

print("Columns Having Missing Values:")
print(missing_summary)


missing_percentage = (
    df.isnull().sum() / len(df) * 100
).sort_values(ascending=False)

missing_percentage = missing_percentage[
    missing_percentage > 0
]

print("Missing Value Percentage:")
print(missing_percentage)


missing_rows = df[df.isnull().any(axis=1)]

print("Rows containing missing values:",
      len(missing_rows))

missing_rows.head(20)


months = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
]

df["CALCULATED_ANNUAL"] = df[months].sum(
    axis=1,
    min_count=12
)

print(df[
    ["YEAR", "ANNUAL", "CALCULATED_ANNUAL"]
].head(10))


df["ANNUAL_DIFFERENCE"] = (
    df["ANNUAL"] - df["CALCULATED_ANNUAL"]
)

annual_check = df[
    df["ANNUAL"].notna() &
    df["CALCULATED_ANNUAL"].notna()
]

print("Rows Checked:", len(annual_check))

print(
    "Maximum Absolute Difference:",
    annual_check["ANNUAL_DIFFERENCE"].abs().max()
)

print(
    "Mean Absolute Difference:",
    annual_check["ANNUAL_DIFFERENCE"].abs().mean()
)


exact_matches = np.isclose(
    df["ANNUAL"],
    df["CALCULATED_ANNUAL"],
    equal_nan=True
).sum()

print("Rows with Exact Annual Match:",
      exact_matches)


df["CALCULATED_JF"] = df[
    ["JAN", "FEB"]
].sum(axis=1, min_count=2)

df["CALCULATED_MAM"] = df[
    ["MAR", "APR", "MAY"]
].sum(axis=1, min_count=3)

df["CALCULATED_JJAS"] = df[
    ["JUN", "JUL", "AUG", "SEP"]
].sum(axis=1, min_count=4)

df["CALCULATED_OND"] = df[
    ["OCT", "NOV", "DEC"]
].sum(axis=1, min_count=3)


print(
    "JF Maximum Difference:",
    (df["JF"] - df["CALCULATED_JF"]).abs().max()
)

print(
    "MAM Maximum Difference:",
    (df["MAM"] - df["CALCULATED_MAM"]).abs().max()
)

print(
    "JJAS Maximum Difference:",
    (df["JJAS"] - df["CALCULATED_JJAS"]).abs().max()
)

print(
    "OND Maximum Difference:",
    (df["OND"] - df["CALCULATED_OND"]).abs().max()
)


temporary_columns = [
    "CALCULATED_ANNUAL",
    "ANNUAL_DIFFERENCE",
    "CALCULATED_JF",
    "CALCULATED_MAM",
    "CALCULATED_JJAS",
    "CALCULATED_OND"
]

df = df.drop(columns=temporary_columns)

print("Final Shape:", df.shape)


subdivision_rainfall = (
    df.groupby("SUBDIVISION")["ANNUAL"]
    .mean()
    .sort_values(ascending=False)
)

print("Top 10 Wettest Subdivisions:")
print(subdivision_rainfall.head(10))

print("\nTop 10 Driest Subdivisions:")
print(subdivision_rainfall.tail(10))

plt.figure(figsize=(12, 8))

subdivision_rainfall.sort_values().plot(
    kind="barh"
)

plt.title(
    "Average Annual Rainfall by IMD Subdivision"
)

plt.xlabel("Average Annual Rainfall (mm)")
plt.ylabel("IMD Subdivision")

plt.tight_layout()
plt.show()

monthly_average = df[months].mean()

print(monthly_average)


plt.figure(figsize=(12, 6))

monthly_average.plot(
    kind="line",
    marker="o"
)

plt.title(
    "Average Monthly Rainfall Across IMD Subdivisions"
)

plt.xlabel("Month")
plt.ylabel("Average Rainfall (mm)")

plt.grid(True)
plt.tight_layout()
plt.show()


seasons = [
    "JF",
    "MAM",
    "JJAS",
    "OND"
]

seasonal_average = df[seasons].mean()

print(seasonal_average)


plt.figure(figsize=(8, 6))

seasonal_average.plot(
    kind="bar"
)

plt.title("Average Seasonal Rainfall")

plt.xlabel("Season")
plt.ylabel("Average Rainfall (mm)")

plt.tight_layout()
plt.show()


variability = (
    df.groupby("SUBDIVISION")["ANNUAL"]
    .agg(["mean", "std", "count"])
)

variability["CV"] = (
    variability["std"] /
    variability["mean"]
) * 100

variability = variability.sort_values(
    "CV",
    ascending=False
)

print("Most Variable Subdivisions:")
print(variability.head(10))

print("\nMost Stable Subdivisions:")
print(variability.tail(10))


plt.figure(figsize=(12, 8))

variability["CV"].sort_values().plot(
    kind="barh"
)

plt.title(
    "Annual Rainfall Variability by IMD Subdivision"
)

plt.xlabel("Coefficient of Variation (%)")
plt.ylabel("IMD Subdivision")

plt.tight_layout()
plt.show()












df = df.sort_values(
    ["SUBDIVISION", "YEAR"]
).reset_index(drop=True)

print(df[["SUBDIVISION", "YEAR", "ANNUAL"]].head(10))


df["PREV_ANNUAL"] = (
    df.groupby("SUBDIVISION")["ANNUAL"]
    .shift(1)
)

df[
    ["SUBDIVISION", "YEAR", "ANNUAL", "PREV_ANNUAL"]
].head(15)


months = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
]

for month in months:
    df[f"PREV_{month}"] = (
        df.groupby("SUBDIVISION")[month]
        .shift(1)
    )

df[
    [
        "SUBDIVISION",
        "YEAR",
        "PREV_JAN",
        "PREV_JUN",
        "PREV_JUL",
        "PREV_DEC"
    ]
].head(10)

seasons = [
    "JF",
    "MAM",
    "JJAS",
    "OND"
]

for season in seasons:
    df[f"PREV_{season}"] = (
        df.groupby("SUBDIVISION")[season]
        .shift(1)
    )
    
df[
    [
        "SUBDIVISION",
        "YEAR",
        "PREV_JF",
        "PREV_MAM",
        "PREV_JJAS",
        "PREV_OND"
    ]
].head(10)

df["ROLLING_3YR_MEAN"] = (
    df.groupby("SUBDIVISION")["ANNUAL"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=3,
            min_periods=3
        ).mean()
    )
)
df[
    [
        "SUBDIVISION",
        "YEAR",
        "ANNUAL",
        "ROLLING_3YR_MEAN"
    ]
].head(10)


df["ROLLING_5YR_MEAN"] = (
    df.groupby("SUBDIVISION")["ANNUAL"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=5,
            min_periods=5
        ).mean()
    )
)
df["RAINFALL_CHANGE"] = (
    df["PREV_ANNUAL"] -
    df.groupby("SUBDIVISION")["ANNUAL"].shift(2)
)
df["RAINFALL_CHANGE_PERCENT"] = (
    df["RAINFALL_CHANGE"] /
    df.groupby("SUBDIVISION")["ANNUAL"].shift(2)
) * 100
df["HISTORICAL_MEAN"] = (
    df.groupby("SUBDIVISION")["ANNUAL"]
    .transform(
        lambda x: x.shift(1).expanding().mean()
    )
)
df["HISTORICAL_STD"] = (
    df.groupby("SUBDIVISION")["ANNUAL"]
    .transform(
        lambda x: x.shift(1).expanding().std()
    )
)

df["RAINFALL_ANOMALY"] = (
    df["ANNUAL"] -
    df["HISTORICAL_MEAN"]
)

df["STANDARDIZED_ANOMALY"] = (
    df["RAINFALL_ANOMALY"] /
    df["HISTORICAL_STD"]
)

df[
    [
        "SUBDIVISION",
        "YEAR",
        "ANNUAL",
        "HISTORICAL_MEAN",
        "HISTORICAL_STD",
        "RAINFALL_ANOMALY",
        "STANDARDIZED_ANOMALY"
    ]
].head(20)


def classify_rainfall(z):

    if pd.isna(z):
        return np.nan

    elif z < -2:
        return "Extreme Dry"

    elif z < -1:
        return "Dry"

    elif z <= 1:
        return "Normal"

    elif z <= 2:
        return "Wet"

    else:
        return "Extreme Wet"
    
df["RAINFALL_CLASS"] = (
    df["STANDARDIZED_ANOMALY"]
    .apply(classify_rainfall)
)

print(
    df["RAINFALL_CLASS"]
    .value_counts()
)
print(
    df["RAINFALL_CLASS"]
    .value_counts(normalize=True) * 100
)


plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="RAINFALL_CLASS"
)

plt.title("Rainfall Class Distribution")

plt.xlabel("Rainfall Class")
plt.ylabel("Number of Observations")

plt.xticks(rotation=20)

plt.tight_layout()
plt.show()


required_columns = [
    "PREV_ANNUAL",
    "ROLLING_3YR_MEAN",
    "ROLLING_5YR_MEAN",
    "RAINFALL_CLASS"
]

df_ml = df.dropna(
    subset=required_columns
).copy()

print("Original rows:", len(df))
print("ML rows:", len(df_ml))

print(
    df_ml[
        [
            "SUBDIVISION",
            "YEAR",
            "PREV_ANNUAL",
            "ROLLING_3YR_MEAN",
            "ROLLING_5YR_MEAN",
            "RAINFALL_CLASS"
        ]
    ].head(20)
)
print(
    df_ml["RAINFALL_CLASS"]
    .value_counts()
)
print(
    "Total Missing Values:",
    df_ml.isnull().sum().sum()
)
missing_ml = df_ml.isnull().sum()

print(
    missing_ml[missing_ml > 0]
)

print("Final ML Dataset Shape:")
print(df_ml.shape)
df_ml.to_csv(
    "rainfall_feature_engineered.csv",
    index=False
)

print(
    "Feature-engineered dataset saved successfully."
)

print("RAINFALL_CLASS exists:", "RAINFALL_CLASS" in df.columns)

print("\nTarget distribution:")
print(df["RAINFALL_CLASS"].value_counts())

print("Minimum Year:", df_ml["YEAR"].min())
print("Maximum Year:", df_ml["YEAR"].max())

print("\nNumber of observations by year:")
print(df_ml.groupby("YEAR").size().head())
print(df_ml.groupby("YEAR").size().tail())
years = sorted(df_ml["YEAR"].unique())

split_index = int(len(years) * 0.80)

train_years = years[:split_index]
test_years = years[split_index:]

print("Training Period:")
print(train_years[0], "to", train_years[-1])

print("\nTesting Period:")
print(test_years[0], "to", test_years[-1])

# ============================================
# PART 3: MACHINE LEARNING DATA PREPARATION
# ============================================

required_columns = [
    "PREV_ANNUAL",
    "ROLLING_3YR_MEAN",
    "ROLLING_5YR_MEAN",
    "RAINFALL_CLASS"
]

df_ml = df.dropna(
    subset=required_columns
).copy()

df_ml = df_ml.sort_values(
    ["YEAR", "SUBDIVISION"]
).reset_index(drop=True)

print("ML Dataset Shape:", df_ml.shape)

print("\nTarget Column Exists:",
      "RAINFALL_CLASS" in df_ml.columns)

print("\nTarget Distribution:")
print(df_ml["RAINFALL_CLASS"].value_counts())


# ============================================
# CHRONOLOGICAL TRAIN-TEST SPLIT
# ============================================

years = sorted(df_ml["YEAR"].unique())

split_index = int(len(years) * 0.80)

train_years = years[:split_index]
test_years = years[split_index:]

print("Training Period:")
print(train_years[0], "to", train_years[-1])

print("\nTesting Period:")
print(test_years[0], "to", test_years[-1])


# Create training and testing datasets

train_data = df_ml[
    df_ml["YEAR"].isin(train_years)
].copy()

test_data = df_ml[
    df_ml["YEAR"].isin(test_years)
].copy()

print("Training Rows:", len(train_data))
print("Testing Rows :", len(test_data))

# ============================================
# VERIFY TARGET
# ============================================

print("RAINFALL_CLASS in df_ml:",
      "RAINFALL_CLASS" in df_ml.columns)

print("RAINFALL_CLASS in train_data:",
      "RAINFALL_CLASS" in train_data.columns)

print("RAINFALL_CLASS in test_data:",
      "RAINFALL_CLASS" in test_data.columns)

# ============================================
# TARGET VARIABLE
# ============================================

target = "RAINFALL_CLASS"

y_train = train_data[target].copy()
y_test = test_data[target].copy()

print("Training Target:")
print(y_train.value_counts())

print("\nTesting Target:")
print(y_test.value_counts())


# ============================================
# FEATURE SELECTION
# ============================================

feature_columns = [
    "SUBDIVISION",
    
    "PREV_ANNUAL",
    
    "PREV_JAN",
    "PREV_FEB",
    "PREV_MAR",
    "PREV_APR",
    "PREV_MAY",
    "PREV_JUN",
    "PREV_JUL",
    "PREV_AUG",
    "PREV_SEP",
    "PREV_OCT",
    "PREV_NOV",
    "PREV_DEC",
    
    "PREV_JF",
    "PREV_MAM",
    "PREV_JJAS",
    "PREV_OND",
    
    "ROLLING_3YR_MEAN",
    "ROLLING_5YR_MEAN",
    
    "RAINFALL_CHANGE",
    "RAINFALL_CHANGE_PERCENT"
]

print("Total Features:", len(feature_columns))

# ============================================
# INPUT FEATURES
# ============================================

X_train = train_data[feature_columns].copy()
X_test = test_data[feature_columns].copy()

print("X_train Shape:", X_train.shape)
print("X_test Shape :", X_test.shape)

print("\nMissing Values in X_train:")
print(X_train.isnull().sum().sum())

print("\nMissing Values in X_test:")
print(X_test.isnull().sum().sum())


# ============================================
# PART 4: DATA PREPROCESSING
# ============================================

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


categorical_features = [
    "SUBDIVISION"
]

numerical_features = [
    col for col in feature_columns
    if col not in categorical_features
]

print("Categorical Features:")
print(categorical_features)

print("\nNumber of Numerical Features:",
      len(numerical_features))


numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

print("Preprocessing Pipeline Created Successfully!")

from sklearn.linear_model import LogisticRegression

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

logistic_model.fit(X_train, y_train)

print("Logistic Regression trained successfully!")

y_pred_lr = logistic_model.predict(X_test)

print("Prediction completed!")

print("\nFirst 20 Predictions:")
print(y_pred_lr[:20])



from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

accuracy_lr = accuracy_score(y_test, y_pred_lr)

precision_lr = precision_score(
    y_test,
    y_pred_lr,
    average="macro",
    zero_division=0
)

recall_lr = recall_score(
    y_test,
    y_pred_lr,
    average="macro",
    zero_division=0
)

f1_lr = f1_score(
    y_test,
    y_pred_lr,
    average="macro",
    zero_division=0
)

print("========== LOGISTIC REGRESSION ==========")

print("Accuracy :", round(accuracy_lr, 4))
print("Precision:", round(precision_lr, 4))
print("Recall   :", round(recall_lr, 4))
print("F1 Score :", round(f1_lr, 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_lr,
    zero_division=0
))
# ============================================
# LOGISTIC REGRESSION CONFUSION MATRIX
# ============================================

from sklearn.metrics import confusion_matrix

class_labels = [
    "Extreme Dry",
    "Dry",
    "Normal",
    "Wet",
    "Extreme Wet"
]

cm_lr = confusion_matrix(
    y_test,
    y_pred_lr,
    labels=class_labels
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm_lr,
    annot=True,
    fmt="d",
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()
plt.show()


# ============================================
# MODEL 2: K-NEAREST NEIGHBORS
# ============================================

from sklearn.neighbors import KNeighborsClassifier

knn_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(
        n_neighbors=5
    ))
])

knn_model.fit(X_train, y_train)

print("KNN trained successfully!")


# ============================================
# MODEL 2: K-NEAREST NEIGHBORS
# ============================================

from sklearn.neighbors import KNeighborsClassifier

knn_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(
        n_neighbors=5
    ))
])

knn_model.fit(X_train, y_train)

print("KNN trained successfully!")

# ============================================
# KNN EVALUATION
# ============================================

y_pred_knn = knn_model.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)

precision_knn = precision_score(
    y_test,
    y_pred_knn,
    average="macro",
    zero_division=0
)

recall_knn = recall_score(
    y_test,
    y_pred_knn,
    average="macro",
    zero_division=0
)

f1_knn = f1_score(
    y_test,
    y_pred_knn,
    average="macro",
    zero_division=0
)

print("========== KNN ==========")

print("Accuracy :", round(accuracy_knn, 4))
print("Precision:", round(precision_knn, 4))
print("Recall   :", round(recall_knn, 4))
print("F1 Score :", round(f1_knn, 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_knn,
    zero_division=0
))

# ============================================
# MODEL 3: NAIVE BAYES
# ============================================

from sklearn.naive_bayes import GaussianNB

naive_bayes_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GaussianNB())
])

naive_bayes_model.fit(X_train, y_train)

print("Naive Bayes trained successfully!")

# ============================================
# NAIVE BAYES EVALUATION
# ============================================

y_pred_nb = naive_bayes_model.predict(X_test)

accuracy_nb = accuracy_score(y_test, y_pred_nb)

precision_nb = precision_score(
    y_test,
    y_pred_nb,
    average="macro",
    zero_division=0
)

recall_nb = recall_score(
    y_test,
    y_pred_nb,
    average="macro",
    zero_division=0
)

f1_nb = f1_score(
    y_test,
    y_pred_nb,
    average="macro",
    zero_division=0
)

print("========== NAIVE BAYES ==========")

print("Accuracy :", round(accuracy_nb, 4))
print("Precision:", round(precision_nb, 4))
print("Recall   :", round(recall_nb, 4))
print("F1 Score :", round(f1_nb, 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_nb,
    zero_division=0
))

# ============================================
# MODEL 4: DECISION TREE
# ============================================

from sklearn.tree import DecisionTreeClassifier

decision_tree_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    ))
])

decision_tree_model.fit(X_train, y_train)

print("Decision Tree trained successfully!")

y_pred_dt = decision_tree_model.predict(X_test)

accuracy_dt = accuracy_score(
    y_test,
    y_pred_dt
)

precision_dt = precision_score(
    y_test,
    y_pred_dt,
    average="macro",
    zero_division=0
)

recall_dt = recall_score(
    y_test,
    y_pred_dt,
    average="macro",
    zero_division=0
)

f1_dt = f1_score(
    y_test,
    y_pred_dt,
    average="macro",
    zero_division=0
)

print("========== DECISION TREE ==========")

print("Accuracy :", round(accuracy_dt, 4))
print("Precision:", round(precision_dt, 4))
print("Recall   :", round(recall_dt, 4))
print("F1 Score :", round(f1_dt, 4))

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred_dt,
    zero_division=0
))

cm_dt = confusion_matrix(
    y_test,
    y_pred_dt,
    labels=class_labels
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm_dt,
    annot=True,
    fmt="d",
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()
plt.show()

# ============================================
# MODEL 5: SUPPORT VECTOR MACHINE
# ============================================

from sklearn.svm import SVC

svm_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(
        kernel="rbf",
        C=1.0,
        class_weight="balanced",
        random_state=42
    ))
])

svm_model.fit(X_train, y_train)

print("SVM trained successfully!")

y_pred_svm = svm_model.predict(X_test)

accuracy_svm = accuracy_score(
    y_test,
    y_pred_svm
)

precision_svm = precision_score(
    y_test,
    y_pred_svm,
    average="macro",
    zero_division=0
)

recall_svm = recall_score(
    y_test,
    y_pred_svm,
    average="macro",
    zero_division=0
)

f1_svm = f1_score(
    y_test,
    y_pred_svm,
    average="macro",
    zero_division=0
)

print("========== SVM ==========")

print("Accuracy :", round(accuracy_svm, 4))
print("Precision:", round(precision_svm, 4))
print("Recall   :", round(recall_svm, 4))
print("F1 Score :", round(f1_svm, 4))

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred_svm,
    zero_division=0
))

cm_svm = confusion_matrix(
    y_test,
    y_pred_svm,
    labels=class_labels
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm_svm,
    annot=True,
    fmt="d",
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.title("SVM - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()
plt.show()



# ============================================
# PART 10: MODEL COMPARISON
# ============================================

model_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "KNN",
        "Naive Bayes",
        "Decision Tree",
        "SVM"
    ],
    
    "Accuracy": [
        accuracy_lr,
        accuracy_knn,
        accuracy_nb,
        accuracy_dt,
        accuracy_svm
    ],
    
    "Precision": [
        precision_lr,
        precision_knn,
        precision_nb,
        precision_dt,
        precision_svm
    ],
    
    "Recall": [
        recall_lr,
        recall_knn,
        recall_nb,
        recall_dt,
        recall_svm
    ],
    
    "F1 Score": [
        f1_lr,
        f1_knn,
        f1_nb,
        f1_dt,
        f1_svm
    ]
})

# Round values
model_comparison = model_comparison.round(4)

print("========== MODEL COMPARISON ==========")
print(model_comparison)

best_model_row = model_comparison.loc[
    model_comparison["F1 Score"].idxmax()
]

print("========== BEST MODEL ==========")
print("Model    :", best_model_row["Model"])
print("Accuracy :", best_model_row["Accuracy"])
print("Precision:", best_model_row["Precision"])
print("Recall   :", best_model_row["Recall"])
print("F1 Score :", best_model_row["F1 Score"])

plt.figure(figsize=(10, 6))

x = np.arange(len(model_comparison))
width = 0.2

plt.bar(
    x - 1.5 * width,
    model_comparison["Accuracy"],
    width,
    label="Accuracy"
)

plt.bar(
    x - 0.5 * width,
    model_comparison["Precision"],
    width,
    label="Precision"
)

plt.bar(
    x + 0.5 * width,
    model_comparison["Recall"],
    width,
    label="Recall"
)

plt.bar(
    x + 1.5 * width,
    model_comparison["F1 Score"],
    width,
    label="F1 Score"
)

plt.xticks(
    x,
    model_comparison["Model"],
    rotation=20
)

plt.ylabel("Score")
plt.title("Comparison of All Machine Learning Models")
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.show()


# ============================================
# PART 11: KNN HYPERPARAMETER TUNING
# ============================================

from sklearn.neighbors import KNeighborsClassifier

k_values = [3, 5, 7, 9, 11, 15, 21]

knn_results = []

for k in k_values:

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(
            n_neighbors=k,
            weights="distance"
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    knn_results.append([
        k,
        accuracy,
        precision,
        recall,
        f1
    ])

knn_tuning_results = pd.DataFrame(
    knn_results,
    columns=[
        "K",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

knn_tuning_results = knn_tuning_results.round(4)

print("========== KNN TUNING RESULTS ==========")
print(knn_tuning_results)


best_knn_row = knn_tuning_results.loc[
    knn_tuning_results["F1 Score"].idxmax()
]

print("========== BEST K VALUE ==========")
print("K         :", int(best_knn_row["K"]))
print("Accuracy  :", best_knn_row["Accuracy"])
print("Precision :", best_knn_row["Precision"])
print("Recall    :", best_knn_row["Recall"])
print("Macro F1  :", best_knn_row["F1 Score"])


plt.figure(figsize=(9, 5))

plt.plot(
    knn_tuning_results["K"],
    knn_tuning_results["F1 Score"],
    marker="o"
)

plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Macro F1 Score")
plt.title("KNN Hyperparameter Tuning")
plt.grid(True)

plt.tight_layout()
plt.show()

# ============================================
# FINAL TUNED KNN MODEL
# ============================================

best_knn_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(
        n_neighbors=3,
        weights="distance"
    ))
])

best_knn_model.fit(X_train, y_train)

print("Final tuned KNN trained successfully!")

y_pred_best_knn = best_knn_model.predict(X_test)

accuracy_best_knn = accuracy_score(
    y_test,
    y_pred_best_knn
)

precision_best_knn = precision_score(
    y_test,
    y_pred_best_knn,
    average="macro",
    zero_division=0
)

recall_best_knn = recall_score(
    y_test,
    y_pred_best_knn,
    average="macro",
    zero_division=0
)

f1_best_knn = f1_score(
    y_test,
    y_pred_best_knn,
    average="macro",
    zero_division=0
)

print("========== FINAL TUNED KNN ==========")
print("Accuracy :", round(accuracy_best_knn, 4))
print("Precision:", round(precision_best_knn, 4))
print("Recall   :", round(recall_best_knn, 4))
print("Macro F1 :", round(f1_best_knn, 4))

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred_best_knn,
    zero_division=0
))

cm_best_knn = confusion_matrix(
    y_test,
    y_pred_best_knn,
    labels=class_labels
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm_best_knn,
    annot=True,
    fmt="d",
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.title("Final Tuned KNN - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()
plt.show()

from sklearn.tree import DecisionTreeClassifier

tree_results = []

max_depth_values = [3, 5, 7, 10, 15, 20, None]

for depth in max_depth_values:

    tree_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=depth,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42
        ))
    ])

    tree_model.fit(X_train, y_train)

    y_pred_tree = tree_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred_tree)
    precision = precision_score(
        y_test,
        y_pred_tree,
        average="macro",
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred_tree,
        average="macro",
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred_tree,
        average="macro",
        zero_division=0
    )

    tree_results.append({
        "Max Depth": depth,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Macro F1": f1
    })

tree_tuning_results = pd.DataFrame(tree_results)

print(tree_tuning_results)

best_tree_row = tree_tuning_results.loc[
    tree_tuning_results["Macro F1"].idxmax()
]

print("========== BEST DECISION TREE ==========")
print(best_tree_row)

plt.figure(figsize=(9, 5))

plt.plot(
    tree_tuning_results["Max Depth"].astype(str),
    tree_tuning_results["Macro F1"],
    marker="o"
)

plt.xlabel("Maximum Depth")
plt.ylabel("Macro F1 Score")
plt.title("Decision Tree Hyperparameter Tuning")
plt.grid(True)

plt.tight_layout()
plt.show()
# ==========================================
# FINAL TUNED DECISION TREE MODEL
# ==========================================

best_dt_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    ))
])

best_dt_model.fit(X_train, y_train)

y_pred_best_dt = best_dt_model.predict(X_test)

accuracy_best_dt = accuracy_score(
    y_test,
    y_pred_best_dt
)

precision_best_dt = precision_score(
    y_test,
    y_pred_best_dt,
    average="macro",
    zero_division=0
)

recall_best_dt = recall_score(
    y_test,
    y_pred_best_dt,
    average="macro",
    zero_division=0
)

f1_best_dt = f1_score(
    y_test,
    y_pred_best_dt,
    average="macro",
    zero_division=0
)

print("========== FINAL TUNED DECISION TREE ==========")
print("Accuracy :", round(accuracy_best_dt, 4))
print("Precision:", round(precision_best_dt, 4))
print("Recall   :", round(recall_best_dt, 4))
print("Macro F1 :", round(f1_best_dt, 4))

from sklearn.svm import SVC

svm_results = []

C_values = [0.1, 1, 10, 100]
kernel_values = ["linear", "rbf"]

for kernel in kernel_values:
    for C in C_values:

        svm_model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", SVC(
                kernel=kernel,
                C=C,
                class_weight="balanced",
                random_state=42
            ))
        ])

        svm_model.fit(X_train, y_train)

        y_pred_svm = svm_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred_svm)

        precision = precision_score(
            y_test,
            y_pred_svm,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred_svm,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred_svm,
            average="macro",
            zero_division=0
        )

        svm_results.append({
            "Kernel": kernel,
            "C": C,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Macro F1": f1
        })

svm_tuning_results = pd.DataFrame(svm_results)

print(svm_tuning_results)


best_svm_row = svm_tuning_results.loc[
    svm_tuning_results["Macro F1"].idxmax()
]

print("========== BEST SVM ==========")
print(best_svm_row)

plt.figure(figsize=(9, 5))

for kernel in kernel_values:

    subset = svm_tuning_results[
        svm_tuning_results["Kernel"] == kernel
    ]

    plt.plot(
        subset["C"].astype(str),
        subset["Macro F1"],
        marker="o",
        label=kernel
    )

plt.xlabel("C Value")
plt.ylabel("Macro F1 Score")
plt.title("SVM Hyperparameter Tuning")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

best_svm_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(
        kernel="rbf",
        C=10,
        class_weight="balanced",
        random_state=42
    ))
])

best_svm_model.fit(X_train, y_train)

print("Final tuned SVM trained successfully!")
y_pred_best_svm = best_svm_model.predict(X_test)

accuracy_best_svm = accuracy_score(
    y_test,
    y_pred_best_svm
)

precision_best_svm = precision_score(
    y_test,
    y_pred_best_svm,
    average="macro",
    zero_division=0
)

recall_best_svm = recall_score(
    y_test,
    y_pred_best_svm,
    average="macro",
    zero_division=0
)

f1_best_svm = f1_score(
    y_test,
    y_pred_best_svm,
    average="macro",
    zero_division=0
)

print("========== FINAL TUNED SVM ==========")
print("Accuracy :", round(accuracy_best_svm, 4))
print("Precision:", round(precision_best_svm, 4))
print("Recall   :", round(recall_best_svm, 4))
print("Macro F1 :", round(f1_best_svm, 4))

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred_best_svm,
    zero_division=0
))

cm_best_svm = confusion_matrix(
    y_test,
    y_pred_best_svm,
    labels=class_labels
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm_best_svm,
    annot=True,
    fmt="d",
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.title("Final Tuned SVM - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()
plt.show()

from sklearn.linear_model import LogisticRegression

C_values = [0.01, 0.1, 1, 10, 100]

lr_results = []

for C in C_values:

    lr_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=C,
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    lr_model.fit(X_train, y_train)

    y_pred_lr_tuned = lr_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred_lr_tuned)

    precision = precision_score(
        y_test,
        y_pred_lr_tuned,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred_lr_tuned,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred_lr_tuned,
        average="macro",
        zero_division=0
    )

    lr_results.append({
        "C": C,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Macro F1": f1
    })

lr_results_df = pd.DataFrame(lr_results)

print("========== LOGISTIC REGRESSION TUNING ==========")
display(lr_results_df.round(4))

# ==========================================
# FINAL TUNED LOGISTIC REGRESSION
# ==========================================

best_lr_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        C=100,
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

best_lr_model.fit(X_train, y_train)

y_pred_best_lr = best_lr_model.predict(X_test)

accuracy_best_lr = accuracy_score(y_test, y_pred_best_lr)
precision_best_lr = precision_score(
    y_test, y_pred_best_lr,
    average="macro",
    zero_division=0
)
recall_best_lr = recall_score(
    y_test, y_pred_best_lr,
    average="macro",
    zero_division=0
)
f1_best_lr = f1_score(
    y_test, y_pred_best_lr,
    average="macro",
    zero_division=0
)

print("========== FINAL TUNED LOGISTIC REGRESSION ==========")
print("Accuracy :", round(accuracy_best_lr, 4))
print("Precision:", round(precision_best_lr, 4))
print("Recall   :", round(recall_best_lr, 4))
print("Macro F1 :", round(f1_best_lr, 4))

# ==========================================
# FINAL TUNED MODEL COMPARISON
# ==========================================

final_model_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "KNN",
        "Naive Bayes",
        "Decision Tree",
        "SVM"
    ],

    "Accuracy": [
        accuracy_best_lr,
        accuracy_best_knn,
        accuracy_nb,
        accuracy_best_dt,
        accuracy_best_svm
    ],

    "Macro Precision": [
        precision_best_lr,
        precision_best_knn,
        precision_nb,
        precision_best_dt,
        precision_best_svm
    ],

    "Macro Recall": [
        recall_best_lr,
        recall_best_knn,
        recall_nb,
        recall_best_dt,
        recall_best_svm
    ],

    "Macro F1": [
        f1_best_lr,
        f1_best_knn,
        f1_nb,
        f1_best_dt,
        f1_best_svm
    ]
})

final_model_comparison = final_model_comparison.sort_values(
    by="Macro F1",
    ascending=False
).reset_index(drop=True)

print("========== FINAL TUNED MODEL COMPARISON ==========")

display(
    final_model_comparison.round(4)
)

# ==========================================
# BEST MODEL IDENTIFICATION
# ==========================================

best_model_name = final_model_comparison.loc[0, "Model"]
best_macro_f1 = final_model_comparison.loc[0, "Macro F1"]

print("===================================")
print("🏆 BEST FINAL MODEL:", best_model_name)
print("🏆 BEST MACRO F1:", round(best_macro_f1, 4))
print("===================================")

# ==========================================
# FINAL EVALUATION - BEST KNN MODEL
# ==========================================

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Predictions from final tuned KNN
y_pred_final = y_pred_best_knn

# Classification Report
print("========== FINAL KNN CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred_final,
        zero_division=0
    )
)

# Confusion Matrix
cm_final = confusion_matrix(
    y_test,
    y_pred_final,
    labels=[
        "Extreme Dry",
        "Dry",
        "Normal",
        "Wet",
        "Extreme Wet"
    ]
)

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm_final,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Extreme Dry",
        "Dry",
        "Normal",
        "Wet",
        "Extreme Wet"
    ],
    yticklabels=[
        "Extreme Dry",
        "Dry",
        "Normal",
        "Wet",
        "Extreme Wet"
    ]
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix - Final KNN Model")

plt.tight_layout()
plt.show()

# ==========================================
# SAMPLE PREDICTION - FINAL KNN MODEL
# ==========================================

# Select one sample from the test dataset
sample_index = 0

sample_input = X_test.iloc[[sample_index]]
actual_class = y_test.iloc[sample_index]

# Predict rainfall class
predicted_class = best_knn_model.predict(sample_input)[0]

# Display result
print("========== SAMPLE PREDICTION ==========")
print("Sample Index       :", sample_index)
print("Actual Rainfall Class    :", actual_class)
print("Predicted Rainfall Class:", predicted_class)

# Check whether prediction is correct
if predicted_class == actual_class:
    print("Prediction Status   : Correct ✅")
else:
    print("Prediction Status   : Incorrect ❌")