
import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/telco_churn_clean.csv")

# Overall churn rate
churn_rate = (df["Churn"] == "Yes").mean() * 100

print(f"Overall Churn Rate: {churn_rate:.2f}%")

# Churn rate by contract type
print("\nCHURN RATE BY CONTRACT:")
contract_churn = df.groupby("Contract")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)
print(contract_churn.round(2))

# Churn rate by internet service
print("\nCHURN RATE BY INTERNET SERVICE:")
internet_churn = df.groupby("InternetService")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)
print(internet_churn.round(2))

# Average monthly charges by churn status
print("\nAVERAGE MONTHLY CHARGES:")
print(df.groupby("Churn")["MonthlyCharges"].mean().round(2))

# Average customer tenure by churn status
print("\nAVERAGE CUSTOMER TENURE:")
print(df.groupby("Churn")["tenure"].mean().round(2))

# Save analysis results
contract_churn.to_csv("data/contract_churn_analysis.csv")
internet_churn.to_csv("data/internet_churn_analysis.csv")

print("\nAnalysis completed successfully!")
