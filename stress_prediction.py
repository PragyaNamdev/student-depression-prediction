import pandas as pd
import numpy as np

df = pd.read_csv("Student Depression Dataset.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())

df["Financial Stress"].fillna(df["Financial Stress"].mean(), inplace = True)
print(df.isnull().sum())

df = df.drop(["id", "City"], axis = 1)
print(df.columns)


# Encoding categorical variables
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
# scikit-learn
df["Gender"] = le.fit_transform(df["Gender"])
df["Profession"] = le.fit_transform(df["Profession"])
df["Sleep Duration"] = le.fit_transform(df["Sleep Duration"])
df["Dietary Habits"] = le.fit_transform(df["Dietary Habits"])
df["Degree"] = le.fit_transform(df["Degree"])
df["Have you ever had suicidal thoughts ?"] = le.fit_transform(df["Have you ever had suicidal thoughts ?"])
df["Family History of Mental Illness"] = le.fit_transform(df["Family History of Mental Illness"])
print(df.head())


# Feature Scaling
from sklearn.model_selection import train_test_split
# Train test split
X = df.drop("Depression", axis=1)
y = df["Depression"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)


# Model Training
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# model create
model = LogisticRegression(max_iter = 1000)
# model train
model.fit(X_train, y_train)

# prediction 
y_pred = model.predict(X_test)
# accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)


# Random Forest Model
from sklearn.ensemble import RandomForestClassifier
# model create
rf_model = RandomForestClassifier(n_estimators = 100, random_state=42)
# model train
rf_model.fit(X_train, y_train)
# prediction
rf_pred = rf_model.predict(X_test)
# accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy: ", rf_accuracy)


# Confusion Matrix (Model Performance Check)
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# Confusion Matrix Generate karo
cm = confusion_matrix(y_test, rf_pred)
print(cm)

# Heatmap Visualization
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Feature Importance
importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(by="Importance", ascending=False)
print(feature_importance)

# Graph
plt.figure(figsize=(8,5))
sns.barplot(x="Importance", y="Feature", data=feature_importance)
plt.title("Feature Importance for Depression Prediction")
plt.show()


# Accuracy Comparison Graph
models = ['Logistic Regression', 'Random Forest']
accuracies = [accuracy, rf_accuracy]

plt.figure(figsize=(6,4))
plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Models")
plt.show()