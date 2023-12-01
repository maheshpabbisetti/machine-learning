import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("Data/futuresale prediction.csv")

# Exploratory data analysis
print(data.head())
print(data.sample(5))
print(data.isnull().sum())

# Visualize relationships using matplotlib
features = ["TV", "Newspaper", "Radio"]
for feature in features:
    plt.scatter(data["Sales"], data[feature], label=feature)
    plt.xlabel("Sales")
    plt.ylabel(feature)
    plt.title(f"Sales vs {feature}")
    plt.legend()
    plt.show()

# Correlation
correlation = data.corr()["Sales"].sort_values(ascending=False)
print(correlation)

# Train-test split
x = data.drop(["Sales"], axis=1).values
y = data["Sales"].values
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Linear Regression
model = LinearRegression().fit(xtrain, ytrain)
print(model.score(xtest, ytest))

# Prediction
new_features = np.array([[230.1, 37.8, 69.2]])
prediction = model.predict(new_features)
print(f"Predicted Sales: {prediction[0]}")

# Plotting predicted values
plt.scatter(data["Sales"], model.predict(x), label="Actual Sales")
plt.scatter(data["Sales"], model.predict(xtrain), label="Train Predictions")
plt.scatter(data["Sales"], model.predict(xtest), label="Test Predictions")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.legend()
plt.show()
