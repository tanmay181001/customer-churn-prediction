
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# STEP 1: Set project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# STEP 2: Find the cleaned dataset
csv_files = [
    p for p in DATA_DIR.rglob("*.csv")
    if "clean" in p.stem.lower()
]

if len(csv_files) != 1:
    raise FileNotFoundError(
        "Expected exactly one cleaned CSV file in data/. "
        f"Found: {[str(p) for p in csv_files]}"
    )

DATA_PATH = csv_files[0]

print("Loading dataset:", DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)

# STEP 3: Prepare target variable
if "Churn" not in df.columns:
    raise ValueError("Churn column not found!")

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

if df["Churn"].isna().any():
    raise ValueError("Unexpected values in Churn column")

# STEP 4: Separate features and target
X = df.drop(columns=["Churn", "customerID"], errors="ignore")
y = df["Churn"]

# STEP 5: Identify column types
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

# STEP 6: Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# STEP 7: Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# STEP 8: Build Logistic Regression model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# STEP 9: Train the model
print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed!")

# STEP 10: Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# STEP 11: Evaluate model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nMODEL PERFORMANCE")
print("---------------------------")
print(f"Accuracy: {accuracy:.2%}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No Churn", "Churn"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# STEP 12: Save trained model
MODEL_PATH = MODEL_DIR / "churn_model.pkl"

joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")
print("Saved at:", MODEL_PATH)

# STEP 13: Save predictions for dashboard
results = X_test.copy()
results["Actual_Churn"] = y_test.values
results["Predicted_Churn"] = y_pred
results["Churn_Probability"] = y_prob

OUTPUT_PATH = MODEL_DIR / "churn_predictions.csv"
results.to_csv(OUTPUT_PATH, index=False)

print("Predictions saved successfully!")
print("Saved at:", OUTPUT_PATH)
print("\nPROJECT 2 MODEL STEP COMPLETED!")
