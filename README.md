# Wine Quality and Purchase Prediction

## Dataset

The dataset contains wine chemical properties with quality scores:

### Features:
- Fixed acidity 
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates 
- Alcohol

### Targets:
1. `quality` (Original target, scale 0-10)
2. `purchased` (Derived: 1 if quality ≥ 6, else 0)

**Source:** [Kaggle Wine Quality Dataset](https://www.kaggle.com/datasets/sh6147782/winequalityred)

## Live Demo

Access the deployed prediction tool:  
🔗 [https://binimolsabu.pythonanywhere.com/](https://binimolsabu.pythonanywhere.com/)

## Python Implementation

```python
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('winequality-red.csv')

# Step 1: Predict quality (original target)
X = data.drop('quality', axis=1)
y = data['quality']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train quality prediction model
quality_model = RandomForestRegressor()
quality_model.fit(X_train, y_train)

# Step 2: Determine purchase decision
def predict_purchase(features):
    """Predicts both quality and purchase decision"""
    quality = quality_model.predict([features])[0]
    purchased = 1 if quality >= 6 else 0
    return quality, purchased

# Save model
pickle.dump(quality_model, open('quality_model.pkl', 'wb'))
