# In a new Python script:
import joblib
import numpy as np

scaler = joblib.load('models/scaler.pkl')
# The mean_ and scale_ (std) arrays will have length 11 (the number of features)
print("Global Mean:", scaler.mean_.tolist())
print("Global Std:", np.sqrt(scaler.var_).tolist())