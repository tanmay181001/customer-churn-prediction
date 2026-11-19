
import pandas as pd

# Step 1: Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Original dataset shape:", df.shape)

# Step 2: Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Step 3: Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Step 4: Remove rows with missing TotalCharges
df = df.dropna(subset=["TotalCharges"]).copy()

# Step 5: Remove duplicate customer IDs
df = df.drop_duplicates(subset=["customerID"])

# Step 6: Standardize Churn values
df["Churn"] = df["Churn"].str.strip()

# Step 7: Save cleaned dataset
df.to_csv("data/telco_churn_clean.csv", index=False)

print("\nCleaned dataset shape:", df.shape)
print("\nCleaned dataset saved successfully!")
print("\nChurn distribution:")
print(df["Churn"].value_counts())
