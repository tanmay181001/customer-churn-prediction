
# Customer Churn Prediction & Retention Analysis

## Project Overview
This project analyzes telecom customer data to identify patterns associated with customer churn and predict which customers are likely to leave a company.

It combines data analysis, machine learning, and business recommendations to support customer retention decisions.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Matplotlib and Seaborn
- VS Code

## Project Workflow
1. Data cleaning and preprocessing
2. Exploratory data analysis
3. Customer churn analysis
4. Machine learning model training
5. Model performance evaluation
6. Interactive Streamlit prediction application
7. Business recommendations for customer retention

## Machine Learning Results

| Metric | Result |
|---|---|
| Model Accuracy | 80.38% |
| ROC-AUC Score | 0.8359 |
| Churn Precision | 0.65 |
| Churn Recall | 0.57 |
| Churn F1-Score | 0.61 |
| Test Records | 1,407 |

The model predicts customer churn risk using customer demographics, account information, and service details.

## Streamlit Application
The interactive application allows users to:
- Enter customer information
- Predict customer churn probability
- View low-risk or high-risk classifications
- Receive business recommendations based on predicted risk

## Business Value
The project demonstrates how predictive analytics can help businesses:
- Identify customers at risk of leaving
- Prioritize customer retention activities
- Understand customer behavior
- Support data-driven business decisions

## Project Structure
- `data/` - Customer datasets and analysis outputs
- `scripts/` - Data cleaning, analysis, and model training
- `models/` - Saved machine learning model and predictions
- `dashboard/` - Streamlit application

## Run the Application

Install dependencies:

```bash
pip install pandas numpy scikit-learn streamlit joblib
```

Start the application from the project root:

```bash
streamlit run dashboard/app.py
```

## Future Improvements
- Add customer segmentation
- Improve churn detection and recall
- Add interactive business dashboards
- Deploy the application online
