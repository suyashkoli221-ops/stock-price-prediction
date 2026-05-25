import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load Dataset
df = pd.read_csv("stock.csv")

# Keep only required columns
df = df[['Open', 'High', 'Low', 'Volume', 'Close']]

# Features and Target
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, 'model.pkl')

print("Model Trained Successfully")