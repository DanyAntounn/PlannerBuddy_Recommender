import nltk
# Ensure NLTK uses Docker-installed data
nltk.data.path.append("/usr/local/nltk_data")

import os, time, json, requests, string
import numpy as np
from openai import OpenAI
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import fuzz

OPENAI_API_KEY = "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"
GOOGLE_API_KEY = "AIzaSyCSWMWno-RiGOWlkswcmoa8Q11VocfPcNs"

sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english") + list(string.punctuation))

def preprocess_text(text: str):
    if not isinstance(text, str):
        return []
    tokens = word_tokenize(text.lower())
    return [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]

# --- Keywords (unchanged) ---
FOOD_CUISINE_KEYWORDS = ["lebanese","mediterranean","bbq","grilled","shawarma","mezza","italian",
    "pasta","pizza","kebab","tabbouleh","hummus","kafta","tawook","sushi","burger",
    "sandwich","dessert","bakery","coffee","tea","armenian"]
FOOD_AMBIANCE_KEYWORDS = ["cozy","quiet","family","friendly","warm","peaceful","romantic","lively",
    "social","trendy","music","party","bustling","vibrant","calm","relaxing","homey","elegant","traditional"]
FOOD_ACTIVITY_KEYWORDS = ["dining","eating","lunch","dinner","breakfast","brunch","snack","takeaway",
    "restaurant","buffet","bar","club","drinks","cafe","pub","lounge"]

ACTIVITY_TYPE_KEYWORDS = ["park","garden","beach","lake","river","mountain","forest","trail","museum",
    "gallery","historical","archaeological","ruins","church","mosque","temple","shop","market",
    "boutique","mall","souvenir","spa","wellness","retreat","yoga","meditation","farm","winery",
    "vineyard","adventure","outdoor","nature","wildlife","scenic","view","waterfall","cultural",
    "art","education","learning","heritage","landmark","family","kids","children","pet-friendly",
    "amusement","theme park","zoo","aquarium","hiking","trekking","walking","exploring","camping",
    "picnic","swimming","boating","kayaking","fishing","watersports","beach day","sightseeing",
    "tour","exhibition","workshop","history","art appreciation","relaxing","unwinding","nightlife",
    "dancing","live music","concert","getaway","vacation","staycation","weekend trip","attraction",
    "guesthouse","hotel","resort","lodge"]
ACTIVITY_AMBIANCE_KEYWORDS = ["cozy","quiet","peaceful","calm","relaxing","serene","lively","social",
    "vibrant","bustling","energetic","fun","trendy","romantic","intimate","charming","elegant",
    "luxurious","family-friendly","kid-friendly","warm","welcoming","friendly","homely","natural",
    "rustic","outdoor","garden","terrace","balcony","view","scenic","modern","traditional","historic",
    "artistic","educational"]

# ========== GPT requirement extractor ==========
client = OpenAI(api_key=OPENAI_API_KEY)
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
            messages=[{"role":"system","content":EXTRACTION_SYSTEM},
                      {"role":"user","content":user_query}],
            temperature=0
        )
        print("OpenAI response:", resp.choices[0].message.content)
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print("OpenAI extraction error:", e)
        return {"requirements": [], "global_location": None}


# ========== Google Places ==========
BASE_URL = "https://maps.googleapis.com/maps/api/place"

class GooglePlacesService:
    def __init__(self, api_key=GOOGLE_API_KEY):
        self.api_key = api_key
        self.session = requests.Session()

    def _details(self, place_id):
        url = f"{BASE_URL}/details/json?place_id={place_id}&fields=name,rating,user_ratings_total,reviews,formatted_address,types&key={self.api_key}"
        try:
            r = self.session.get(url, timeout=20)
            data = r.json()
            if data.get("status")!="OK": return None
            res = data["result"]
            reviews = [rev["text"] for rev in res.get("reviews",[])]
            return {"name":res.get("name"),"address":res.get("formatted_address"),
                    "rating":res.get("rating"),"user_ratings_total":res.get("user_ratings_total"),
                    "reviews":reviews,"types":res.get("types",[])}
        except: return None

    def text_search(self, query, location_str, place_type=None, max_pages=1):
        all_results, token, page = [], None, 0
        while page<max_pages:
            url = f"{BASE_URL}/textsearch/json?query={requests.utils.quote(query+' in '+location_str)}&key={self.api_key}"
            if place_type: url += f"&type={place_type}"
            if token: url += f"&pagetoken={token}"; time.sleep(2)
            data = self.session.get(url).json()
            if data.get("status")!="OK": break
            for p in data["results"]:
                detail = self._details(p["place_id"])
                if detail: all_results.append(detail)
            token=data.get("next_page_token"); page+=1
            if not token: break
        return all_results

places_service=GooglePlacesService()

# ========== NLP scoring ==========
def fuzzy_in(text, needle, thresh=85):
    return fuzz.partial_ratio(text.lower(), needle.lower())>=thresh

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
    primary_found = pick_matches(primary_keywords)
    secondary_found = pick_matches(secondary_keywords)
    ambiance_found = pick_matches(ambiance_keywords)
    sentiments = [sia.polarity_scores(r)["compound"] for r in reviews if isinstance(r, str) and r.strip()]
    avg_sentiment = float(np.mean(sentiments)) if sentiments else 0.0
    return {
        "name": place_dict.get("name", "Unknown"),
        "avg_sentiment": avg_sentiment,
        "primary_features": primary_found,
        "secondary_features": secondary_found,
        "ambiance": ambiance_found,
        "rating": float(rating),
        "num_reviews": int(num_reviews),
        "address": place_dict.get("address", place_dict.get("formatted_address", "N/A")),
        "types": place_dict.get("types", []),
        "raw": place_dict
    }

def build_flutter_payload(plan_steps):
    out = []
    for step in plan_steps:
        req = step["requirement"]
        places = []
        for rec in step["recommendations"]:
            places.append({
                "name": rec["name"],
                "score": round(float(rec["score"]), 2),
                "rating": float(rec["profile"]["rating"] or 0.0),
            })
        out.append({
            "requirement": req,
            "places": places
        })
    return out

def match_user_to_place(profile, place_profile, typ):
    score=0
    if any(f in place_profile["primary_features"] for f in profile["preferred_primary_features"]): score+=5
    score+=len(set(profile["ambiance"])&set(place_profile["ambiance"]))*3
    score+=len(set(profile["preferred_secondary_features"])&set(place_profile["secondary_features"]))*2
    if place_profile["avg_sentiment"]>0.3: score+=3
    if place_profile["rating"]>=profile["rating_threshold"]: score+=4
    if place_profile["num_reviews"]>=profile["min_reviews_count"]: score+=2
    return score

def rank_places(profile, places, typ, top_n, query):
    q_tokens=preprocess_text(query)
    scored=[]
    for p in places:
        pp=extract_place_profile(p,typ)
        s=match_user_to_place(profile,pp,typ)
        if any(qt in pp["primary_features"]+pp["secondary_features"] for qt in q_tokens): s+=4
        scored.append({"name":pp["name"],"score":s,"profile":pp})
    return sorted(scored,key=lambda x:x["score"],reverse=True)[:top_n]

# ========== Planner ==========
def auto_trip_generate(user_food, user_act, location="Beirut, Lebanon"):
    plan = []
    food_kw = " ".join(user_food.get("preferred_primary_features", [])[:2]) or "restaurant"
    raw_food = places_service.text_search(food_kw, location, place_type="restaurant")
    ranked_food = rank_places(user_food, raw_food, "food", top_n=1, query=food_kw)
    plan.append({
        "requirement": {"type": "restaurant", "count": 1, "keywords": [food_kw], "location_hint": location},
        "recommendations": ranked_food
    })
    act_kw = " ".join(user_act.get("preferred_primary_features", [])[:3]) or "attraction"
    raw_act = places_service.text_search(act_kw, location, place_type=None)
    ranked_act = rank_places(user_act, raw_act, "activity", top_n=2, query=act_kw)
    plan.append({
        "requirement": {"type": "activity", "count": 2, "keywords": [act_kw], "location_hint": location},
        "recommendations": ranked_act
    })
    return {"plan": plan}

def custom_trip_generate(query,user_food,user_act,location="Beirut, Lebanon"):
    extraction=extract_requirements_with_openai(query)
    plan=[]
    for req in extraction["requirements"]:
        typ=req["type"]; kw=" ".join(req["keywords"]) or typ
        count=req.get("count",1); loc=req.get("location_hint") or extraction.get("global_location") or location
        raw=places_service.text_search(kw,loc,"restaurant" if typ=="restaurant" else None)
        ranked=rank_places(user_food if typ=="restaurant" else user_act,raw,typ,count,kw)
        plan.append({"requirement":req,"recommendations":ranked})
    return {"extracted":extraction,"plan":plan}
