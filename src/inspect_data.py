
import pandas as pd

# 1. Load the CSV dataset
file_path = "data/raw/Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

# 2. Display the dataset size
print("\n--- DATASET SHAPE ---")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 3. Display the first five records
print("\n--- FIRST FIVE ROWS ---")
print(df.head())

# 4. Display column names and data types
print("\n--- COLUMN INFORMATION ---")
print(df.info())

# 5. Check missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# 6. Check duplicate records
print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

# 7. Examine the target variable
print("\n--- CHURN DISTRIBUTION ---")
print(df["Churn"].value_counts())

print("\n--- CHURN PERCENTAGES ---")
print(df["Churn"].value_counts(normalize=True) * 100)