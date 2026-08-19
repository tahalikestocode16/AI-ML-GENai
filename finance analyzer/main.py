import numpy as np
import pandas as pd

# inspect data

main_df = pd.read_csv("data/transactions.csv")
df = main_df.copy()


print(df.head(5))
# print(df.isnull().sum())
# print(df.duplicated().sum())
# print(df["id"].duplicated())


# print("\n")

# print(df.shape)
# print("\n")
# print(df.info())
# print(df.dtypes)
# print(df.describe())

# =========================
# Overall spending
# =========================

total_spending = df["amount"].sum()
average_spending = df["amount"].mean()
min_spending = df["amount"].min()
max_spending = df["amount"].max()

print("\n--- Overall Spending ---")
print("Total spending:", total_spending)
print("Average spending:", average_spending)
print("Minimum spending:", min_spending)
print("Maximum spending:", max_spending)


# =========================
# Category analysis
# =========================

# Total spending for each category
spending_by_category = df.groupby("category")["amount"].sum()

# Average transaction for each category
average_spending_category = df.groupby("category")["amount"].mean()

# Category with the highest total spending
top_category = spending_by_category.idxmax()

# Category with the lowest total spending
lowest_category = spending_by_category.idxmin()

# Highest average transaction category
highest_avg_category = average_spending_category.idxmax()

# Lowest average transaction category
lowest_avg_category = average_spending_category.idxmin()

# Percentage of total spending per category
category_percentage = (spending_by_category / total_spending) * 100


print("\n--- Spending by Category ---")
print(spending_by_category)

print("\n--- Average Spending by Category ---")
print(average_spending_category)

print("\n--- Category Summary ---")
print("Top spending category:", top_category)
print("Lowest spending category:", lowest_category)
print("Highest average transaction category:", highest_avg_category)
print("Lowest average transaction category:", lowest_avg_category)

print("\n--- Percentage of Total Spending ---")
print(category_percentage)

# print(top_spending_category)
# print(average_spending_category)


# =========================
# Date analysis
# =========================

print(df["date"].head(5))
df["date"].to_datetime()
print(df["date"].dtype)
