# Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Loading the  train and test data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

train_data = train_data.dropna(subset=features + [target])

X_train = train_data[features]
y_train = train_data[target]

model = LinearRegression()
model.fit(X_train, y_train)

print("\nMissing values in test data:")
print(test_data[features].isnull().sum())

test_data[features] = test_data[features].fillna(test_data[features].median())

X_test = test_data[features]
test_predictions = model.predict(X_test)

submission = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': test_predictions
})

# Saveing submission file
submission.to_csv('submission.csv', index=False)
print("\nSubmission file 'submission.csv' created successfully.")

y_train_pred = model.predict(X_train)
mse = mean_squared_error(y_train, y_train_pred)
r2 = r2_score(y_train, y_train_pred)

print("\nModel Performance on Training Data:")
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_train, y=y_train_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices (Training Data)')
plt.show()