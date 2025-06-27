import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Step 1: Load and preprocess dataset
df = pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')

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

# Label Encoding
le = LabelEncoder()
df['Fuel_Type'] = le.fit_transform(df['Fuel_Type'])
df['Seller_Type'] = le.fit_transform(df['Seller_Type'])
df['Transmission'] = le.fit_transform(df['Transmission'])
df['Owner'] = le.fit_transform(df['Owner'])

# Car age
df['Car_Age'] = 2025 - df['Year']
df.drop(['Year', 'Car_Name'], axis=1, inplace=True)

# Features and target
X = df.drop(['Selling_Price'], axis=1)
y = df['Selling_Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# GUI Setup
root = tk.Tk()
root.title("🚗 Car Price Predictor")
root.geometry("500x600")
root.config(bg="#f0f8ff")

# Title Label
tk.Label(root, text="Car Price Predictor", font=("Helvetica", 20, "bold"),
         bg="#4682B4", fg="white", pady=10).pack(fill="x")

# Input Fields
def make_label_input(text):
    tk.Label(root, text=text, font=("Arial", 12, "bold"), bg="#f0f8ff").pack(pady=(10, 0))
    entry = tk.Entry(root, font=("Arial", 12))
    entry.pack()
    return entry

kms_entry = make_label_input("Kilometers Driven")
year_entry = make_label_input("Year of Purchase")

# Dropdowns
fuel_options = {"Petrol": 2, "Diesel": 0, "CNG": 1}
seller_options = {"Individual": 1, "Dealer": 0, "Trustmark Dealer": 2}
trans_options = {"Manual": 1, "Automatic": 0}
owner_options = {"First Owner": 0, "Second Owner": 1, "Third Owner": 2, "Fourth & Above Owner": 3, "Test Drive Car": 4}

def make_dropdown(label, options):
    tk.Label(root, text=label, font=("Arial", 12, "bold"), bg="#f0f8ff").pack(pady=(10, 0))
    cb = ttk.Combobox(root, values=list(options.keys()), font=("Arial", 12))
    cb.pack()
    return cb

fuel_cb = make_dropdown("Fuel Type", fuel_options)
seller_cb = make_dropdown("Seller Type", seller_options)
trans_cb = make_dropdown("Transmission", trans_options)
owner_cb = make_dropdown("Owner Type", owner_options)

# Prediction
def predict_price():
    try:
        kms = int(kms_entry.get())
        year = int(year_entry.get())
        age = 2025 - year
        fuel = fuel_options[fuel_cb.get()]
        seller = seller_options[seller_cb.get()]
        trans = trans_options[trans_cb.get()]
        owner = owner_options[owner_cb.get()]

        input_data = np.array([[kms, fuel, seller, trans, owner, age]])
        predicted = model.predict(input_data)[0]

        messagebox.showinfo("Prediction", f"Estimated Price: ₹ {predicted:,.0f}")
    except:
        messagebox.showerror("Error", "Please fill all fields correctly!")

# Predict Button
tk.Button(root, text="Predict Price", font=("Arial", 14, "bold"),
          bg="#4682B4", fg="white", command=predict_price).pack(pady=20)

root.mainloop()
