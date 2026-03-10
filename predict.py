import joblib
import numpy as np

# load model
model = joblib.load("restaurant_rating_model.pkl")

# example input
sample = np.array([[2,10,500,1,1,3,150]])

prediction = model.predict(sample)

print("Predicted Rating Category:",prediction)