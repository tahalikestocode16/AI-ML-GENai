import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


main_df = pd.read_csv("data/emails.csv") 
df = main_df.copy()

print(df.head())
print(df.info())
print(df.shape)
# print(df.isnull().sum())
# print(df.describe())

print(df["label"].value_counts()) # this tells how many times each label appears 

X = df["message"]
y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
# assigned numerica values to each word in the sentences to make finding spam easier
X_test = vectorizer.transform(X_test)


model = LogisticRegression()
prediction = model.fit(X_train, y_train).predict(X_test)
print(prediction)

# mse wont work here since we arent dealing with numbers 

accuracy = accuracy_score(y_test, prediction)
matrix = confusion_matrix(y_test, prediction)
target_names = ["ham", "spam"]
report = classification_report(y_test, prediction, target_names= target_names)

print("overall model report: ==================================\n")
print(f'accuracy: {accuracy}')
print(f'confusion matrix \n {matrix}')
print(f'classification_report: \n {report}')
# weighted avg gives every class importance based on how many there are if there are 50 male 20 female males would have advantage
# macro avg would go pound for pound


# taking emails from user
email_address = input("Enter your email address: ").lower().strip()

while True:
# not using it just to give the program a nicer feel
    message = input("Enter message: ") 
    
    if(message == "q" or message == "Q"):
        print("Program ended")
        break
    
    print("classifying...")
    
# now we vectorize the users message so our model can understand
    message = vectorizer.transform([message])

    prediction_message = model.predict(message)
    print(prediction_message)
  