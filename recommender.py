import nltk
# Ensure NLTK uses Docker-installed data
nltk.data.path.append("/usr/local/nltk_data")

import os, time, json, requests, string, base64
import numpy as np
from openai import OpenAI
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import fuzz
from google.cloud import firestore
from google.oauth2 import service_account
from google.cloud.firestore_v1 import GeoPoint

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# -------- Firestore client setup using Render Secret File --------
def get_firestore_client():
    # Path where Render mounts the secret file
    secret_path = "/etc/secrets/service-account.json"

    if os.path.exists(secret_path):
        with open(secret_path, "r") as f:
            info = json.load(f)

        creds = service_account.Credentials.from_service_account_info(info)
        return firestore.Client(credentials=creds, project=info["project_id"])

    # Fallback for local dev if you set GOOGLE_APPLICATION_CREDENTIALS
    return firestore.Client()

db = get_firestore_client()

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
        except:
            return None

    def text_search(self, query, location_str, place_type=None, max_pages=1):
        all_results, token, page = [], None, 0
        while page<max_pages:
            url = f"{BASE_URL}/textsearch/json?query={requests.utils.quote(query+' in '+location_str)}&key={self.api_key}"
            if place_type:
                url += f"&type={place_type}"
            if token:
                url += f"&pagetoken={token}"
                time.sleep(2)
            data = self.session.get(url).json()
            if data.get("status")!="OK":
                break
            for p in data["results"]:
                detail = self._details(p["place_id"])
                if detail:
                    all_results.append(detail)
            token = data.get("next_page_token")
            page += 1
            if not token:
                break
        return all_results

places_service = GooglePlacesService()

# ========== NLP scoring ==========
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

    primary_found = pick_matches(primary_keywords)
    secondary_found = pick_matches(secondary_keywords)
    ambiance_found = pick_matches(ambiance_keywords)

    sentiments = [
        sia.polarity_scores(r)["compound"]
        for r in reviews
        if isinstance(r, str) and r.strip()
    ]
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
            profile = rec["profile"]

            place_entry = {
                "name": rec["name"],
                "score": round(float(rec["score"]), 2),
                "rating": float(profile["rating"] or 0.0),
            }

            # NEW: include lat/long from Firestore profile if available
            lat = profile.get("latitude")
            lon = profile.get("longitude")
            if lat is not None and lon is not None:
                place_entry["latitude"] = lat
                place_entry["longitude"] = lon

            places.append(place_entry)

        out.append({
            "requirement": req,
            "places": places
        })
    return out



def match_user_to_place(profile, place_profile, typ):
    score = 0
    if any(f in place_profile["primary_features"] for f in profile["preferred_primary_features"]):
        score += 5
    score += len(set(profile["ambiance"]) & set(place_profile["ambiance"])) * 3
    score += len(set(profile["preferred_secondary_features"]) & set(place_profile["secondary_features"])) * 2
    if place_profile["avg_sentiment"] > 0.3:
        score += 3
    if place_profile["rating"] >= profile["rating_threshold"]:
        score += 4
    if place_profile["num_reviews"] >= profile["min_reviews_count"]:
        score += 2
    return score

def rank_places(profile, places, typ, top_n, query):
    q_tokens = preprocess_text(query)
    scored = []
    for p in places:
        pp = extract_place_profile(p, typ)
        s = match_user_to_place(profile, pp, typ)
        if any(qt in pp["primary_features"] + pp["secondary_features"] for qt in q_tokens):
            s += 4
        scored.append({"name": pp["name"], "score": s, "profile": pp})
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_n]

# ========== Planner (Google Places based) ==========
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

def custom_trip_generate(query, user_food, user_act, location="Beirut, Lebanon"):
    extraction = extract_requirements_with_openai(query)
    plan = []
    for req in extraction["requirements"]:
        typ = req["type"]
        kw = " ".join(req["keywords"]) or typ
        count = req.get("count", 1)
        loc = req.get("location_hint") or extraction.get("global_location") or location
        raw = places_service.text_search(kw, loc, "restaurant" if typ == "restaurant" else None)
        ranked = rank_places(user_food if typ == "restaurant" else user_act, raw, typ, count, kw)
        plan.append({"requirement": req, "recommendations": ranked})
    return {"extracted": extraction, "plan": plan}

# ========== Firestore enrichment (already used) ==========
def build_place_dict_from_firestore_doc(doc_data: dict) -> dict:
    """
    Convert a Firestore restaurant/activity doc into the dict shape
    expected by extract_place_profile (name, rating, user_ratings_total, reviews, types, address).
    """
    reviews = doc_data.get("reviews") or []
    rating = doc_data.get("rating", 0.0) or 0.0
    num_reviews = doc_data.get("num_reviews") or len(reviews)

    return {
        "name": doc_data.get("name", "Unknown"),
        "address": doc_data.get("address", "N/A"),
        "rating": rating,
        "user_ratings_total": num_reviews,
        "reviews": reviews,
        "types": doc_data.get("types", []),
    }

def enrich_collection_with_profiles_from_firestore(collection_name: str, recommendation_type: str):
    """
    Enrich each document in a Firestore collection (RestaurantsFinal / ActivitiesFinal)
    using ONLY the reviews that are already stored in Firestore.

    recommendation_type: "food" for restaurants, "activity" for activities.
    """
    print(f"=== Enriching collection: {collection_name} (type={recommendation_type}) ===")

    docs = db.collection(collection_name).stream()
    processed = 0
    skipped_no_reviews = 0

    for snap in docs:
        doc_id = snap.id
        data = snap.to_dict() or {}

        # Skip if there are no reviews stored at all
        reviews = data.get("reviews") or []
        if not reviews:
            skipped_no_reviews += 1
            continue

        # Optional: skip if already enriched (idempotent)
        if "avg_sentiment" in data and "primary_features" in data:
            continue

        print(f"\n[Doc {doc_id}] name={data.get('name')} - {len(reviews)} reviews")

        # Build a fake "place_dict" as if it came from Google Places,
        # but using only the Firestore fields.
        place_dict = build_place_dict_from_firestore_doc(data)

        # Use your EXISTING extractor
        profile = extract_place_profile(place_dict, recommendation_type)

        update_data = {
            "avg_sentiment": profile["avg_sentiment"],
            "primary_features": profile["primary_features"],
            "secondary_features": profile["secondary_features"],
            "ambiance": profile["ambiance"],
            "rating": profile["rating"],
            "num_reviews": profile["num_reviews"],
            "address": profile["address"],
            "types": profile["types"],
            # keep reviews field as is (already in Firestore)
        }

        db.collection(collection_name).document(doc_id).set(update_data, merge=True)
        processed += 1
        print(f"   -> Updated Firestore doc {doc_id} with NLP profile.")

    print("\n=== Summary for", collection_name, "===")
    print("Processed docs:", processed)
    print("Skipped (no reviews):", skipped_no_reviews)
    print("====================================\n")

def build_profile_from_firestore_doc(doc_data: dict, recommendation_type: str):
    """
    Build a place_profile in the same shape as extract_place_profile(),
    but using fields already stored in Firestore after enrichment.
    Also extracts latitude/longitude from the `location` GeoPoint if present.
    """
    reviews = doc_data.get("reviews") or []
    rating = float(doc_data.get("rating", 0.0) or 0.0)
    num_reviews = int(doc_data.get("num_reviews") or len(reviews))

    # Extract GeoPoint from Firestore "location" field
    latitude = None
    longitude = None
    loc = doc_data.get("location")

    # Firestore GeoPoint object (google.cloud.firestore_v1._helpers.GeoPoint)
    if loc is not None:
        # Robust: support GeoPoint object or dict
        if hasattr(loc, "latitude") and hasattr(loc, "longitude"):
            latitude = loc.latitude
            longitude = loc.longitude
        elif isinstance(loc, dict):
            # In case it was stored manually as a dict
            latitude = loc.get("latitude")
            longitude = loc.get("longitude")

    return {
        "name": doc_data.get("name", "Unknown"),
        "avg_sentiment": float(doc_data.get("avg_sentiment", 0.0) or 0.0),
        "primary_features": doc_data.get("primary_features", []),
        "secondary_features": doc_data.get("secondary_features", []),
        "ambiance": doc_data.get("ambiance", []),
        "rating": rating,
        "num_reviews": num_reviews,
        "address": doc_data.get("address", "N/A"),
        "types": doc_data.get("types", []),
        "latitude": latitude,
        "longitude": longitude,
        "raw": doc_data,
    }



def rank_places_from_firestore(
    collection_name: str,
    user_profile: dict,
    recommendation_type: str,
    top_n: int,
    query: str = ""
):
    """
    Read all docs from `collection_name`, build place_profile from stored fields,
    score them against user_profile, and return the top_n.
    """
    docs = db.collection(collection_name).stream()
    q_tokens = preprocess_text(query)
    scored = []

    for snap in docs:
        data = snap.to_dict() or {}

        # Only consider docs that were enriched
        if "primary_features" not in data or "avg_sentiment" not in data:
            continue

        place_profile = build_profile_from_firestore_doc(data, recommendation_type)
        s = match_user_to_place(user_profile, place_profile, recommendation_type)

        # Optional query boost (similar to rank_places)
        if q_tokens:
            combined_features = place_profile["primary_features"] + place_profile["secondary_features"]
            if any(qt in combined_features for qt in q_tokens):
                s += 4

        scored.append({
            "name": place_profile["name"],
            "score": float(s),
            "profile": place_profile,
            "doc_id": snap.id,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]

def firestore_trip_generate(
    user_food: dict,
    user_act: dict,
    num_restaurants: int = 1,
    num_activities: int = 2,
    location: str = "Beirut, Lebanon",
    user_latitude: float | None = None,
    user_longitude: float | None = None,
):
    """
    Use ONLY Firestore-enriched data (RestaurantsFinal, ActivitiesFinal)
    to pick the best restaurants and activities for the given user profiles.

    user_latitude / user_longitude are accepted for future distance-based logic.
    """
    plan = []

    # Restaurants
    food_kw = " ".join(user_food.get("preferred_primary_features", [])[:2]) or "restaurant"
    top_restaurants = rank_places_from_firestore(
        collection_name="RestaurantsFinal",
        user_profile=user_food,
        recommendation_type="food",
        top_n=num_restaurants,
        query=food_kw,
    )
    plan.append({
        "requirement": {
            "type": "restaurant",
            "count": num_restaurants,
            "keywords": [food_kw],
            "location_hint": location,
        },
        "recommendations": top_restaurants,
    })

    # Activities
    act_kw = " ".join(user_act.get("preferred_primary_features", [])[:3]) or "activity"
    top_activities = rank_places_from_firestore(
        collection_name="ActivitiesFinal",
        user_profile=user_act,
        recommendation_type="activity",
        top_n=num_activities,
        query=act_kw,
    )
    plan.append({
        "requirement": {
            "type": "activity",
            "count": num_activities,
            "keywords": [act_kw],
            "location_hint": location,
        },
        "recommendations": top_activities,
    })

    # For now we ignore user_latitude / user_longitude,
    # but they are available here when you want to add distance scoring.
    return {"plan": plan}

