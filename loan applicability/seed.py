import pandas as pd
import random

random.seed(42)

data = []

education_options = ["Graduate", "Not Graduate"]
self_employed_options = ["Yes", "No"]
married_options = ["Yes", "No"]
property_options = ["Urban", "Rural", "Semiurban"]

for i in range(200):

    income = random.randint(2000, 12000)
    loan_amount = random.randint(50000, 350000)
    loan_term = random.choice([120, 180, 240, 300, 360])
    credit_history = random.choice([0, 1])
    education = random.choice(education_options)
    self_employed = random.choice(self_employed_options)
    married = random.choice(married_options)
    dependents = random.choice([0, 1, 2, 3])
    property_area = random.choice(property_options)

    # Simple rules to create a realistic target
    score = 0

    if income >= 5000:
        score += 2

    if loan_amount <= income * 25:
        score += 2

    if credit_history == 1:
        score += 3

    if education == "Graduate":
        score += 1

    if self_employed == "No":
        score += 1

    if dependents <= 2:
        score += 1

    if property_area == "Semiurban":
        score += 1

    if score >= 6:
        loan_status = "Yes"
    else:
        loan_status = "No"

    data.append([
        income,
        loan_amount,
        loan_term,
        credit_history,
        education,
        self_employed,
        married,
        dependents,
        property_area,
        loan_status
    ])

columns = [
    "income",
    "loan_amount",
    "loan_term",
    "credit_history",
    "education",
    "self_employed",
    "married",
    "dependents",
    "property_area",
    "loan_status"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("data/loan_data.csv", index=False)

print(df.head())
print(df.shape)
print(df["loan_status"].value_counts())