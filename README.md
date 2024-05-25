- 👋 Hi, I’m @Kashishbhattt
- 👀 I’m interested in computer science
- 🌱 I’m currently learning ai using python
- <--
- my project tpic is:Data Collection and Preprocessing
- -->
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('equipment_data.csv')

# Preprocess data
data.dropna(inplace=True)  # Drop missing values
X = data.drop('failure', axis=1)  # Features
y = data['failure']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
