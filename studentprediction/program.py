import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

main_df = pd.read_csv("data/students.csv")
df = main_df.copy()

print(df.head(5))
print(df.isnull().sum())
print(df.info())
print(df.shape)
print(df.describe())

df = df.dropna()

X = df[[
    "study_hours",
    "attendance",
    "previous_score",
    "assignment_score",
    "sleep_hours"
]]

y = df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, # 0.2 means 20 percent of the data is for testing here
    random_state=42
)

model = LinearRegression()

predictions = model.fit(X_train, y_train).predict(X_test)

print(predictions)

study_hours = float(input("Study hours: "))
attendance = float(input("Attendance: "))
previous_score = float(input("Previous score: "))
assignment_score = float(input("Assignment score: "))
sleep_hours = float(input("Sleep hours: "))

new_student = [[
    study_hours,
    attendance,
    previous_score,
    assignment_score,
    sleep_hours
]]

prediction = model.predict(new_student)

print(f"Predicted final score: {prediction[0]:.2f}")

mse = mean_squared_error(y_test, predictions)
print("mae", mse)
rmse = np.sqrt(mse)
print("rmse", rmse)



# model coefficients 

for feature, coefficient in zip(X.columns, model.coef_):
    print(f"{feature}: {coefficient:.2f}")

print(f"X intercept: {model.intercept_:.2f}")