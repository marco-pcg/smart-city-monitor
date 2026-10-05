from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

encoder = SentenceTransformer('all-MiniLM-L6-v2')

reports = [
    "There is a massive pothole on Main Street.",
    "The trash hasn't been collected in two weeks.",
    "Someone is breaking into a car on 5th Ave.",
    "The streetlight is out on Oak Lane.",
    "Illegal dumping in the park behind the school.",
    "I saw a suspicious person looking into windows."
]

# 0 = Roads, 1 = Sanitation, 2 = Police
labels = [0, 1, 2, 0, 1, 2]

embeddings = encoder.encode(reports)
print(f"Embedding shape: {embeddings.shape}") # (6, 384)

X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.3)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

new_report = ["The garbage truck missed my street again."]
new_embedding = encoder.encode(new_report)
prediction = classifier.predict(new_embedding)

departments = {0: "Roads", 1: "Sanitation", 2: "Police"}

print(f"New report: '{new_report[0]}'")
print(f"Routed to: {departments[prediction[0]]}")

import json, joblib
joblib.dump(classifier, "models/04/report_classifier.pkl")

label_map = {0: "Roads", 1: "Sanitation", 2: "Police"}
with open("models/04/label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)

encoder_config = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_dim": 384,
    "num_classes": len(label_map)
}
with open("models/04/encoder_config.json", "w") as f:
    json.dump(encoder_config, f, indent=2)
