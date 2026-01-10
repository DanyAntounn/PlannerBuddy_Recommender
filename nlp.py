# services/nlp.py
import nltk
nltk.data.path.append("/usr/local/nltk_data")

import json, string
import numpy as np
from openai import OpenAI
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import fuzz
import os

try:
    wn.ensure_loaded()
except Exception:
    _ = wn.synsets("dog")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english") + list(string.punctuation))

# -------- Keywords --------
FOOD_CUISINE_KEYWORDS = [
    "lebanese","mediterranean","bbq","grilled","shawarma","mezza","italian",
    "pasta","pizza","kebab","tabbouleh","hummus","kafta","tawook","sushi","burger",
    "sandwich","dessert","bakery","coffee","tea","armenian"
]

FOOD_AMBIANCE_KEYWORDS = [
    "cozy","quiet","family","friendly","warm","peaceful","romantic","lively",
    "social","trendy","music","party","bustling","vibrant","calm","relaxing",
    "homey","elegant","traditional"
]

FOOD_ACTIVITY_KEYWORDS = [
    "dining","eating","lunch","dinner","breakfast","brunch","snack","takeaway",
    "restaurant","buffet","bar","club","drinks","cafe","pub","lounge"
]

ACTIVITY_TYPE_KEYWORDS = [
    "park","garden","beach","lake","river","mountain","forest","trail","museum",
    "gallery","historical","archaeological","ruins","church","mosque","temple",
    "shop","market","boutique","mall","souvenir","spa","wellness","retreat",
    "yoga","meditation","farm","winery","vineyard","adventure","outdoor","nature",
    "wildlife","scenic","view","waterfall","cultural","art","education","learning",
    "heritage","landmark","family","kids","children","pet-friendly","amusement",
    "theme park","zoo","aquarium","hiking","trekking","walking","exploring",
    "camping","picnic","swimming","boating","kayaking","fishing","watersports",
    "beach day","sightseeing","tour","exhibition","workshop","history",
    "art appreciation","relaxing","unwinding","nightlife","dancing","live music",
    "concert","getaway","vacation","staycation","weekend trip","attraction",
    "guesthouse","hotel","resort","lodge"
]

ACTIVITY_AMBIANCE_KEYWORDS = [
    "cozy","quiet","peaceful","calm","relaxing","serene","lively","social",
    "vibrant","bustling","energetic","fun","trendy","romantic","intimate",
    "charming","elegant","luxurious","family-friendly","kid-friendly","warm",
    "welcoming","friendly","homely","natural","rustic","outdoor","garden",
    "terrace","balcony","view","scenic","modern","traditional","historic",
    "artistic","educational"
]

# -------- Helpers --------
def preprocess_text(text: str):
    if not isinstance(text, str):
        return []
    tokens = word_tokenize(text.lower())
    out = []
    for t in tokens:
        if t.isalpha() and t not in stop_words:
            try:
                out.append(lemmatizer.lemmatize(t))
            except Exception:
                out.append(t)
    return out

def fuzzy_in(text, needle, thresh=85):
    return fuzz.partial_ratio(text.lower(), needle.lower()) >= thresh

def extract_place_profile(place_dict, recommendation_type: str):
    reviews = place_dict.get("reviews", []) or []
    rating = place_dict.get("rating", 0.0) or 0.0
    num_reviews = place_dict.get("user_ratings_total", 0) or 0

    review_blob = " ".join([r for r in reviews if isinstance(r, str)])[:20000]
    tokens = preprocess_text(review_blob)

    if recommendation_type == "food":
        primary_keywords = FOOD_CUISINE_KEYWORDS
        secondary_keywords = FOOD_ACTIVITY_KEYWORDS
        ambiance_keywords = FOOD_AMBIANCE_KEYWORDS
    else:
        primary_keywords = ACTIVITY_TYPE_KEYWORDS
        secondary_keywords = ACTIVITY_TYPE_KEYWORDS
        ambiance_keywords = ACTIVITY_AMBIANCE_KEYWORDS

    def pick_matches(candidates):
        found = set()
        for kw in candidates:
            if kw in tokens or fuzzy_in(review_blob, kw, 88):
                found.add(kw)
        return list(found)

    sentiments = [
        sia.polarity_scores(r)["compound"]
        for r in reviews
        if isinstance(r, str) and r.strip()
    ]

    return {
        "name": place_dict.get("name", "Unknown"),
        "avg_sentiment": float(np.mean(sentiments)) if sentiments else 0.0,
        "primary_features": pick_matches(primary_keywords),
        "secondary_features": pick_matches(secondary_keywords),
        "ambiance": pick_matches(ambiance_keywords),
        "rating": float(rating),
        "num_reviews": int(num_reviews),
        "address": place_dict.get("address", "N/A"),
        "types": place_dict.get("types", []),
        "raw": place_dict,
    }

# -------- GPT extraction --------
EXTRACTION_SYSTEM = """You are a precise extractor.
Input: a natural-language trip query.
Output: ONLY JSON:
{
  "requirements":[
    {"type":"restaurant"|"activity","count":int,"keywords":[...],"location_hint":str|null,"time_hint":str|null}
  ],
  "global_location": str|null
}"""

def extract_requirements_with_openai(user_query: str) -> dict:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": user_query}
            ],
            temperature=0
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {"requirements": [], "global_location": None}
