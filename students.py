import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("data.csv").head(5)

# making a duplicate to work on it so original data doesnt get ruined

df_label = df.copy()
df_minmax = df.copy()
df_standard = df.copy()

# Create scalers
minmax = MinMaxScaler()
# minmax scaler is used when you want 0-1 range and are sure some values dont dominate
standard = StandardScaler()
# calculaets dro the mean and average so even if a value i tlarge it gets put in a fair decimal
# mostly used and correct me i fim wrong

le = LabelEncoder()

df_label["Passed_Encoded"] = le.fit_transform(df_label["Passed"])
# df_label["StudyHhours_Encoded"] = le.fit_transform(df_label["StudyHours"])
df_label["Name_encoded"] = le.fit_transform(df_label["Name"])

print(df_label[['Passed', "Passed_Encoded", "Name", "Name_encoded"]])

df_minmax["Studyhours_scaled"] = minmax.fit_transform(
    
    df_minmax[["StudyHours"]]
    
    )
df_standard["Studyhours_standardscaled"] = standard.fit_transform(df_standard[["StudyHours"]])

print("\n data scaling using minmax and standard scaling")
print(df_standard)
print("\n")
print(df_minmax)