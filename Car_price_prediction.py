# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load Dataset
df = pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')
print("✅ Data Loaded Successfully!")
print(df.head())

# Step 3: Rename columns for consistency (optional but helpful)
df.rename(columns={
    'name': 'Car_Name',
    'year': 'Year',
    'selling_price': 'Selling_Price',
    'km_driven': 'Kms_Driven',
    'fuel': 'Fuel_Type',
    'seller_type': 'Seller_Type',
    'transmission': 'Transmission',
    'owner': 'Owner'
}, inplace=True)

# Step 4: Data Preprocessing
le = LabelEncoder()
df['Fuel_Type'] = le.fit_transform(df['Fuel_Type'])
df['Seller_Type'] = le.fit_transform(df['Seller_Type'])
df['Transmission'] = le.fit_transform(df['Transmission'])
df['Owner'] = le.fit_transform(df['Owner'])  #Add this line to fix error

# Create new feature for car age
df['Car_Age'] = 2025 - df['Year']
df.drop(['Year'], axis=1, inplace=True)

# Drop Car_Name if it exists
if 'Car_Name' in df.columns:
    df.drop(['Car_Name'], axis=1, inplace=True)


# Step 5: Prepare features and target
X = df.drop(['Selling_Price'], axis=1)
y = df['Selling_Price']

# Step 6: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Predict
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print(f"MAE (Mean Absolute Error): {mae:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 10: Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x=model.feature_importances_, y=X.columns)
plt.rcParams['font.family'] = 'Segoe UI Emoji'  # Windows emoji supported font
plt.title("🔍 Feature Importance")

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()



