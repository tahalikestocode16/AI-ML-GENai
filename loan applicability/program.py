import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# ================= DATA =================

main_df = pd.read_csv("data/loan_data.csv")
df = main_df.copy()

print(df.head())

# print(df.info())
# print(df.shape)
# print(df.isnull().sum())
# print(df.describe())


# ================= X AND Y =================

X = df.drop("loan_status", axis=1)
y = df["loan_status"]


# ================= TRAIN TEST SPLIT =================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ================= COLUMNS =================

categorical_columns = [
    "education",
    "self_employed",
    "married",
    "property_area"
]

numerical_columns = [
    "income",
    "loan_amount",
    "loan_term",
    "credit_history",
    "dependents"
]


# ================= PREPROCESSING =================

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(), categorical_columns)
    ],
    remainder="passthrough"
)


X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


# ================= MODEL =================

model = LogisticRegression()

model.fit(X_train, y_train)

prediction = model.predict(X_test)

# print(prediction)


# ================= LOAN PROGRAM =================

print("\n================= Loan Eligibility Program =================")


while True:

    name = input("Enter your name or press Q to quit: ").strip().lower()
    if(name == "q"):
        break
    

    # Name is not used by the model.
    # It is only for the user experience.


    # ================= EDUCATION =================

    while True:

        education = input(
            "Enter your education (Graduate/Not Graduate): "
        ).strip().title()

        if education in ["Graduate", "Not Graduate"]:
            break

        print("Invalid input. Please enter Graduate or Not Graduate.")


    # ================= SELF EMPLOYED =================

    while True:

        self_employed = input(
            "Are you self employed? (Yes/No): "
        ).strip().title()

        if self_employed in ["Yes", "No"]:
            break

        print("Invalid input. Please enter Yes or No.")


    # ================= MARRIED =================

    while True:

        married = input(
            "Are you married? (Yes/No): "
        ).strip().title()

        if married in ["Yes", "No"]:
            break

        print("Invalid input. Please enter Yes or No.")


    # ================= PROPERTY AREA =================

    while True:

        property_area = input(
            "Enter your property area (Semiurban/Rural/Urban): "
        ).strip().title()

        if property_area in ["Semiurban", "Rural", "Urban"]:
            break

        print("Invalid input. Please enter Semiurban, Rural or Urban.")


    # ================= INCOME =================

    while True:

        try:

            income = float(
                input("Enter your income: ")
            )

            break

        except ValueError:

            print("Invalid input. Please enter a valid number.")


    # ================= LOAN AMOUNT =================

    while True:

        try:

            loan_amount = float(
                input("Enter loan amount: ")
            )

            break

        except ValueError:

            print("Invalid input. Please enter a valid number.")


    # ================= LOAN TERM =================

    while True:

        try:

            loan_term = int(
                input("Enter loan term in months (280/140/120): ")
            )

            if loan_term in [280, 140, 120]:
                break

            print("Please enter 280, 140 or 120.")

        except ValueError:

            print("Invalid input. Please enter a valid number.")


    # ================= CREDIT HISTORY =================

    while True:

        try:

            credit_history = int(
                input("Do you have credit history? (0 = No / 1 = Yes): ")
            )

            if credit_history in [0, 1]:
                break

            print("Please enter 0 or 1.")

        except ValueError:

            print("Invalid input. Please enter 0 or 1.")


    # ================= DEPENDENTS =================

    while True:

        try:

            dependents = int(
                input("How many people are you in charge of? ")
            )

            if dependents >= 0:
                break

            print("Dependents cannot be negative.")

        except ValueError:

            print("Invalid input. Please enter a valid number.")


    # ================= CREATE USER DATA =================

    user_data = pd.DataFrame([{

        "income": income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "credit_history": credit_history,
        "education": education,
        "self_employed": self_employed,
        "married": married,
        "dependents": dependents,
        "property_area": property_area

    }])


    # ================= PREPROCESS USER DATA =================

    user_data = preprocessor.transform(user_data)


    # ================= PREDICTION =================

    user_prediction = model.predict(user_data)

    print("\n================ RESULT ================")
    print(f"Name: {name}")
    print(f"Loan status: {user_prediction[0]}")
    print("========================================\n")