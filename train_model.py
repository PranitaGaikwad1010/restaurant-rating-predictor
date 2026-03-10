import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Load Dataset
# -------------------------------

df = pd.read_csv("Dataset.csv")

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

# -------------------------------
# Data Cleaning
# -------------------------------

# Convert Yes/No columns to 1/0
yes_no_columns = [
    "Has Table booking",
    "Has Online delivery",
    "Is delivering now",
    "Switch to order menu"
]

for col in yes_no_columns:
    if col in df.columns:
        df[col] = df[col].map({"Yes": 1, "No": 0})

# -------------------------------
# Encode Categorical Data
# -------------------------------

city_encoder = LabelEncoder()
cuisine_encoder = LabelEncoder()

df["City"] = city_encoder.fit_transform(df["City"].astype(str))
df["Cuisines"] = cuisine_encoder.fit_transform(df["Cuisines"].astype(str))

# -------------------------------
# Feature Selection
# -------------------------------

features = [
    "City",
    "Cuisines",
    "Average Cost for two",
    "Price range",
    "Votes",
    "Has Table booking",
    "Has Online delivery"
]

target = "Aggregate rating"

X = df[features]
y = df[target]

# -------------------------------
# Train Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Model
# -------------------------------

model = RandomForestRegressor(n_estimators=200, random_state=42)

model.fit(X_train, y_train)

# -------------------------------
# Predictions
# -------------------------------

y_pred = model.predict(X_test)

# -------------------------------
# Model Evaluation
# -------------------------------

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)

# -------------------------------
# Visualization
# -------------------------------

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Rating")
plt.show()

# -------------------------------
# Save Model
# -------------------------------

joblib.dump(model, "restaurant_model.pkl")
joblib.dump(city_encoder, "city_encoder.pkl")
joblib.dump(cuisine_encoder, "cuisine_encoder.pkl")

print("Model and encoders saved successfully!")