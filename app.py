import pandas as pd
import pickle
import joblib
from flask import Flask, request, jsonify, render_template
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load dataset
df = pd.read_csv("Dataset.csv")

# Convert 'Login Time' and 'Logout Time' to datetime format
df["Login Time"] = pd.to_datetime(df["Login Time"], errors='coerce')
df["Logout Time"] = pd.to_datetime(df["Logout Time"], errors='coerce')

# Drop rows with invalid datetime conversion
df.dropna(subset=["Login Time", "Logout Time"], inplace=True)

# Extract hour from 'Login Time'
df["Login Hour"] = df["Login Time"].dt.hour

# Feature Engineering
df["Survey Attempt Rate"] = df["Count of Survey Attempts"] / (df["Usage Time (mins)"] + 1)

# Compute active hour count
active_hours = df.groupby("Login Hour")["NPI"].count().rename("Active Hour Count")

# Load trained model
with open("saved_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load label encoders
encoders = joblib.load("encoders.pkl")

@app.route('/')
def home():
    return render_template('index.html')  # Load UI

@app.route('/get_best_doctors', methods=['POST'])
def get_best_doctors():
    try:
        # Get time input from form
        time_input = request.form.get("time")

        if not time_input:
            return jsonify({"error": "Time input is required (format: HH:MM)"}), 400

        # Convert input time
        time_input = datetime.strptime(time_input, "%H:%M").time()

        # Filter active doctors
        active_doctors = df[
            (df["Login Time"].dt.time <= time_input) & (df["Logout Time"].dt.time >= time_input)
        ].copy()

        if active_doctors.empty:
            return jsonify({"message": f"No active doctors found at {time_input}."}), 200

        # Recompute feature engineering for active doctors
        active_doctors["Login Hour"] = active_doctors["Login Time"].dt.hour
        active_doctors["Survey Attempt Rate"] = active_doctors["Count of Survey Attempts"] / (active_doctors["Usage Time (mins)"] + 1)

        # Merge with active hour count
        active_doctors = active_doctors.merge(active_hours, on="Login Hour", how="left")

        # Encode categorical columns using the saved label encoders
        for col in ["State", "Region", "Speciality"]:
            if col in active_doctors.columns:
                active_doctors[col] = active_doctors[col].map(lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1)

        # Prepare data for prediction
        X_active = active_doctors.drop(columns=["NPI", "Count of Survey Attempts", "Login Time", "Logout Time"])

        # Predict survey probability
        active_doctors["Survey Probability"] = model.predict_proba(X_active)[:, 1]

        # Select top 10 doctors
        best_doctors = active_doctors.nlargest(10, "Survey Probability")[["NPI", "Survey Probability"]]

        return jsonify(best_doctors.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
