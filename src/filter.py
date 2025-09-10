import pandas as pd

# Load your original dataset
df = pd.read_csv(r"/home/maddy-das/myproject/data/customer_data.csv")

# Select only the columns you want
selected_columns = [
    "customerID",       # or "CustomerID" depending on your file
    "SeniorCitizen",
    "gender",           # or "Gender"
    "tenure",           # or "Tenure"
    "MonthlyCharges",
    "Contract",         # or "ContractType"
    "InternetService",
    "TechSupport",
    "TotalCharges",
    "Churn"
]

# Keep only the available ones (ignore missing)
filtered_df = df[[col for col in selected_columns if col in df.columns]]

# Save to a new CSV
filtered_df.to_csv("filtered_dataset.csv", index=False)

print("Filtered dataset saved as filtered_dataset.csv")
