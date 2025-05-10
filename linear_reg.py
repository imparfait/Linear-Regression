import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("energy_usage.csv")
print("First 5 rows of the dataset:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
X = df[["temperature", "humidity", "hour", "is_weekend"]]
y = df["consumption"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue')
plt.xlabel("Actual Consumption")
plt.ylabel("Predicted Consumption)")
plt.title("Actual vs Predicted Consumption")
plt.grid(True)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--') 
plt.tight_layout()
plt.show()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mean_consumption = y_test.mean()
error_percentage = (mae / mean_consumption) * 100
print(f"\nMAE: {mae:.2f} kWh")
print(f"RMSE: {rmse:.2f} kWh")
print(f"Average percentage error: {error_percentage:.2f}%")