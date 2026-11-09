import joblib

model = joblib.load("model.joblib")
print("model loaded from model.joblib")

flower = [[5.1, 3.5, 1.4, 0.2]]

pridiction = model.predict(flower)
print("pridiction = ", pridiction)

print("Notice what happened: no training, no data loading. It just loaded the file and predicted. That’s exactly how a deployed model works in production.")