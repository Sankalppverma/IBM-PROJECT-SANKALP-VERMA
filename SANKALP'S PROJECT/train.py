import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv("train.csv")

# Select features
features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "FullBath",
    "BedroomAbvGr",
    "YearBuilt"
]

X = data[features]
y = data["SalePrice"]

# Train model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

# Save model
joblib.dump(model, "house_model.pkl")

print("✅ Better model saved successfully!")