import pandas as pd

churn_df = pd.read_csv("Churn_modeling_train_test.csv")
bank_df = pd.read_csv("bank-full_train_test.csv")

print("Churn dataset:")
print(churn_df.head())
print(churn_df.shape)

print("\nBank dataset:")
print(bank_df.head())
print(bank_df.shape)