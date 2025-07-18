from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
data = load_iris()
X, y = data.data, data.target

# Train model
clf = RandomForestClassifier()
clf.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)
