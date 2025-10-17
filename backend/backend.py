from flask import Flask, request, jsonify 
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
data = pd.read_csv('smishingDB_augmented.csv', encoding='latin-1')
X_train, X_test, y_train, y_test = train_test_split(data['TEXT'], data['LABEL'], test_size=0.2, random_state=42)
spam_detector_pipeline = Pipeline([
    ('words_to_vector', TfidfVectorizer(ngram_range=(1, 2))),
    ('classifier', RandomForestClassifier())
])
spam_detector_pipeline.fit(X_train, y_train)

def getSuspiciousWords(message, pipeline, top_n=5):
    vectorizer = pipeline.named_steps['words_to_vector']
    message_to_vector_matrix = vectorizer.transform([message])
    feature_array = np.array(vectorizer.get_feature_names_out())
    words_to_vector_scores = message_to_vector_matrix.toarray().flatten()
    top_Suspicious = words_to_vector_scores.argsort()[-top_n:][::-1]
    Suspicious_words = feature_array[top_Suspicious]
    return Suspicious_words.tolist()

def SaveSpamMessage(message_from_user, label):
    try:
        existing_data = pd.read_csv('smishingDB_augmented.csv', encoding='latin-1')
        if message_from_user in existing_data['TEXT'].values:
            print("Message is exists.")
            return
    except FileNotFoundError:
        pass  

    new_data = pd.DataFrame({'LABEL': [label], 'TEXT': [message_from_user]})
    new_data.to_csv('smishingDB_augmented.csv', mode='a', header=False, index=False, encoding='latin-1')

@app.route('/check', methods=['POST'])
def check_message():
    data = request.json
    message_from_user = data.get('message', '')
    suspicious_words = None

    prediction = spam_detector_pipeline.predict([message_from_user])[0]
    if prediction == 0:
        print("Result: Not Spam\n")
        SaveSpamMessage(message_from_user, 0)
        suspicious_words = []
    else:
        print("Result: Spam\n")
        suspicious_words = getSuspiciousWords(message_from_user, spam_detector_pipeline)
        SaveSpamMessage(message_from_user, 1)
    response = {
        'result': int(prediction),
        'suspicious_words': suspicious_words
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=8000)