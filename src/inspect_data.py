import pandas as pd
df = pd.read_csv("dataset/heart.csv")
print("\n====DATASET SHAPE====")
print(df.shape)

print("\n====FIRST 5 ROW====")
print(df.head)

print("\n====COLUMN NAMES====")
print(df.columns.tolist())

print("\n====MISSING VAL====")
print(df.isnull().sum())

print("\n====DUP VAL====")
print(df.duplicated().sum())

print("\n====SUMMARY===")
print(df.describe())

print("\n====TARGET DISTRI===")
print(df.iloc[:,-1].value_counts())