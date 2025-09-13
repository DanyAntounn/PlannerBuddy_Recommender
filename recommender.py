import nltk
# Ensure NLTK uses Docker-installed data
nltk.data.path.append("/usr/local/nltk_data")
import os


import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer

# Initialize tools
stop_words = set(stopwords.words("english"))
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

# Text preprocessing
def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(w) for w in words if w.isalpha() and w not in stop_words]
    return words

# Simple scoring
def compute_score(user_profile, features, reviews):
    score = 0
    # Match features
    for pref in user_profile.get("preferences", []):
        if pref.lower() in features:
            score += 1
    # Sentiment on reviews
    sentiments = [sia.polarity_scores(r)["compound"] for r in reviews]
    if sentiments:
        score += float(np.mean(sentiments))
    return float(score)

# Food recommender
def recommend_food_places(user_profile, food_places):
    ranked = []
    for place in food_places:
        features = preprocess_text(place.get("description", ""))
        score = compute_score(user_profile, features, place.get("reviews", []))
        ranked.append({"name": place["name"], "score": score})
    return sorted(ranked, key=lambda x: x["score"], reverse=True)

# Activity recommender
def recommend_activity_places(user_profile, activity_places):
    ranked = []
    for place in activity_places:
        features = preprocess_text(place.get("description", ""))
        score = compute_score(user_profile, features, place.get("reviews", []))
        ranked.append({"name": place["name"], "score": score})
    return sorted(ranked, key=lambda x: x["score"], reverse=True)
