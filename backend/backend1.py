from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import time
import pandas as pd
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

data = pd.read_csv('smishingDB_augmented.csv', encoding='latin-1')
texts = data['TEXT'].astype(str)
labels = data['LABEL'].astype(int)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

linear_svc = LinearSVC()
linear_svc.fit(X, labels)

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X, labels)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X, labels)

def save_spam_message(message_from_user, sum):
    file_path = 'smishingDB_augmented.csv'
    message_in_data = ""

    if os.path.exists(file_path):
        existing_data = pd.read_csv(file_path, encoding='latin-1')
        if message_from_user in existing_data['TEXT'].values:
            print("Message already exists in dataset.")
            message_in_data = "Message already exists in dataset."
            return message_in_data
    else:
        existing_data = pd.DataFrame(columns=['LABEL', 'TEXT'])
        message_in_data = "Message added to dataset."

    if sum >= 2:
        new_data = pd.DataFrame({'LABEL': [1], 'TEXT': [message_from_user]})
    else:
        new_data = pd.DataFrame({'LABEL': [0], 'TEXT': [message_from_user]})

    new_data.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False, encoding='latin-1')
    message_in_data = "Message added to dataset."
    return message_in_data

def get_top_suspicious_words(text, vectorizer, max_words = 5):
    response = vectorizer.transform([text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]
    top_five_words = feature_array[tfidf_sorting][:max_words]
    return top_five_words.tolist()

@app.route("/checkSMS", methods=["POST"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    data = request.json
    input_text = data.get("text", "")

    if not input_text.strip():
        return jsonify({"error": "Empty message"}), 400

    X_input = vectorizer.transform([input_text])

    results = {}
    sum_spam = 0 
    Post_classification_label = 0 
    suspicious_words = [] 

    start = time.time()
    svc_pred = linear_svc.predict(X_input)[0]
    if svc_pred == 1:
        sum_spam += 1
    results["LinearSVC"] = {"prediction": int(svc_pred), "time": round((time.time() - start) * 1000, 2)}

    start = time.time()
    rf_pred = random_forest.predict(X_input)[0]
    if rf_pred == 1:
        sum_spam += 1
    results["RandomForest"] = {"prediction": int(rf_pred), "time": round((time.time() - start) * 1000, 2)}

    start = time.time()
    xgb_pred = xgb_model.predict(X_input)[0]
    if xgb_pred == 1:
        sum_spam += 1
    results["XGBoost"] = {"prediction": int(xgb_pred), "time": round((time.time() - start) * 1000, 2)}

    message_in_data = save_spam_message(input_text, sum_spam)
    
    if sum_spam >= 2:
        Post_classification_label = 1

    if Post_classification_label == 1:
        suspicious_words = get_top_suspicious_words(input_text, vectorizer)
    else:
        suspicious_words = []

    return jsonify({
        "results": results,
        "final_prediction": int(Post_classification_label),
        "suspicious_words": suspicious_words,
        "message_info": message_in_data
    })

if __name__ == "__main__":
    app.run(debug=True)
