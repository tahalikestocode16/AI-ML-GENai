import pandas as pd
import numpy as np
import matplotlib.pyplot as plt





main_df = pd.read_csv("data/transactions.csv")
df = main_df.copy()

print(df.info())
print(df.shape)
print(df.describe())

print(df.isnull().sum())


# ==============================
# Overall Spending
# ==============================

total_spending = df["amount"].sum()
average_spending = df["amount"].mean()
max_spending = df["amount"].max()
minimum_spending = df["amount"].min()

print("\n" + "=" * 50)
print("           FINANCE ANALYZER")
print("=" * 50)

print("\n--- Overall Spending ---")
print(f"Total spending:      {total_spending:,.2f}")
print(f"Average spending:    {average_spending:,.2f}")
print(f"Minimum spending:    {minimum_spending:,.2f}")
print(f"Maximum spending:    {max_spending:,.2f}")


# ==============================
# Spending by Category
# ==============================

total_spending_category = df.groupby("category")["amount"].sum()

highest_category = total_spending_category.idxmax()
lowest_category = total_spending_category.idxmin()

average_spending_category = df.groupby("category")["amount"].mean()

highest_avg_category = average_spending_category.idxmax()
lowest_avg_category = average_spending_category.idxmin()

spending_percentage = (total_spending_category / total_spending) * 100

transactions_count_category = df.groupby("category").size()

highest_transaction_category = transactions_count_category.idxmax()
lowest_transaction_category = transactions_count_category.idxmin()


print("\n--- Spending by Category ---")
print(total_spending_category)

print("\n--- Average Spending by Category ---")
print(average_spending_category)

print("\n--- Category Summary ---")
print(f"Top spending category:          {highest_category}")
print(f"Lowest spending category:       {lowest_category}")
print(f"Highest average transaction:    {highest_avg_category}")
print(f"Lowest average transaction:     {lowest_avg_category}")
print(f"Most transactions category:     {highest_transaction_category}")
print(f"Least transactions category:    {lowest_transaction_category}")

print("\n--- Percentage of Total Spending ---")
print(spending_percentage.round(2))


# ==============================
# Monthly Spending
# ==============================

df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month

monthly_spending = df.groupby("month")["amount"].sum()

highest_sales_month = monthly_spending.idxmax()
lowest_sales_month = monthly_spending.idxmin()

average_monthly_sales = monthly_spending.mean()

print("\n--- Monthly Spending ---")
print(monthly_spending)

print("\n--- Monthly Summary ---")
print(f"Highest spending month:    {highest_sales_month}")
print(f"Lowest spending month:     {lowest_sales_month}")
print(f"Average monthly spending:  {average_monthly_sales:,.2f}")
# says to add comma and 2 decimal places round off


# ==============================
# Payment Methods
# ==============================

payment_method = df.groupby("payment_method")["amount"].sum()

most_used_paymentmethod = payment_method.idxmax()
least_used_paymentmethod = payment_method.idxmin()

print("\n--- Spending by Payment Method ---")
print(payment_method)

print("\n--- Payment Method Summary ---")
print(f"Most spending through:     {most_used_paymentmethod}")
print(f"Least spending through:    {least_used_paymentmethod}")


# ==============================
# Location Data
# ==============================

location_sales = df.groupby("location")["amount"].sum()

highest_sales_location = location_sales.idxmax()
lowest_sales_location = location_sales.idxmin()

print("\n--- Spending by Location ---")
print(location_sales)

print("\n--- Location Summary ---")
print(f"Highest spending location:  {highest_sales_location}")
print(f"Lowest spending location:   {lowest_sales_location}")


# ==============================
# Money Spent On
# ==============================

total_spent_on = df.groupby("description")["amount"].sum()

most_spent_on = total_spent_on.idxmax()
least_spent_on = total_spent_on.idxmin()

print("\n--- Spending by Description ---")
print(total_spent_on)

print("\n--- Description Summary ---")
print(f"Most spent on:              {most_spent_on}")
print(f"Least spent on:             {least_spent_on}")


# ==============================
# End
# ==============================

print("\n" + "=" * 50)
print("             END OF REPORT")
print("=" * 50)

# Bar chart ======================================================

# plt.figure()
# X = total_spending_category.index
# y = total_spending_category.values
# plt.bar(X,y)

# plt.title("Total Spending by Category")
# plt.xlabel("Category")
# plt.ylabel("Amount Spent")

# plt.xticks(rotation=45)
# # rotates so they dont overlap on eachother
# for i, value in enumerate(total_spending_category.values):
#     plt.text(i, value, f"{value:,.0f}", ha="center", va="bottom")
# plt.grid(axis="y", linestyle="--", alpha=0.4)
# plt.tight_layout()
# plt.show()

# Pie chart
# plt.figure()
# plt.pie(
#     spending_percentage.values,
#     labels=spending_percentage.index
# )

# plt.tight_layout()
# plt.title("Spending Distribution by Category")
# plt.show()

# figure chart 

plt.plot(
    monthly_spending.index,
    monthly_spending.values,
    marker="o"
)

plt.title("Monthly Spending")
plt.xlabel("Month")
plt.ylabel("Amount Spent")

plt.grid(axis="y", linestyle="--", alpha=0.4)

for i, value in enumerate(monthly_spending.values):
    plt.text(i, value, f"{value:,.0f}", ha="center", va="bottom")

plt.tight_layout()
plt.show()


# payment methods chart 
plt.figure()

plt.bar(
    payment_method.index,
    payment_method.values
)

plt.title("Spending by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Amount Spent")

for i, value in enumerate(payment_method.values):
    plt.text(i, value, f"{value:,.0f}", ha="center", va="bottom")

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()


# Spending by Location

plt.figure()

plt.bar(
    location_sales.index,
    location_sales.values
)

plt.title("Spending by Location")
plt.xlabel("Location")
plt.ylabel("Amount Spent")

plt.xticks(rotation=45)

for i, value in enumerate(location_sales.values):
    plt.text(i, value, f"{value:,.0f}", ha="center", va="bottom")

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()