# recommender.py
import nltk
nltk.data.path.append("/usr/local/nltk_data")

import os, time, json, requests, string, math
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
import random

from nltk.corpus import wordnet as wn

try:
    wn.ensure_loaded()
except Exception:
    _ = wn.synsets("dog")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# IMPORTANT for option 1:
# This is your API's public base URL (Render env var).
# Example: https://plannerbuddy-api.onrender.com
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

# -------- Firestore client setup using Render Secret File --------
def get_firestore_client():
    secret_path = "/etc/secrets/service-account.json"
    if os.path.exists(secret_path):
        with open(secret_path, "r") as f:
            info = json.load(f)
        creds = service_account.Credentials.from_service_account_info(info)
        return firestore.Client(credentials=creds, project=info["project_id"])
    return firestore.Client()

db = get_firestore_client()

sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english") + list(string.punctuation))

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
        url = (
            f"{BASE_URL}/details/json"
            f"?place_id={place_id}"
            f"&fields=place_id,name,rating,user_ratings_total,reviews,formatted_address,types,photos,geometry"
            f"&key={self.api_key}"
        )
        try:
            r = self.session.get(url, timeout=20)
            data = r.json()
            if data.get("status") != "OK":
                return None

            res = data["result"]
            reviews = [rev["text"] for rev in res.get("reviews", [])]

            photos = res.get("photos", []) or []
            photo_ref = photos[0].get("photo_reference") if photos else None

            geom = res.get("geometry", {}).get("location", {}) or {}
            lat = geom.get("lat")
            lng = geom.get("lng")

            return {
                "place_id": res.get("place_id", place_id),
                "name": res.get("name"),
                "address": res.get("formatted_address"),
                "rating": res.get("rating"),
                "user_ratings_total": res.get("user_ratings_total"),
                "reviews": reviews,
                "types": res.get("types", []),
                "photo_reference": photo_ref,
                "latitude": lat,
                "longitude": lng,
            }
        except:
            return None

    def text_search(self, query, location_str, place_type=None, max_pages=1):
        all_results, token, page = [], None, 0
        while page < max_pages:
            url = f"{BASE_URL}/textsearch/json?query={requests.utils.quote(query + ' in ' + location_str + ', Lebanon')}&key={self.api_key}"
            if place_type:
                url += f"&type={place_type}"
            if token:
                url += f"&pagetoken={token}"
                time.sleep(2)
            data = self.session.get(url).json()
            if data.get("status") != "OK":
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

# =========================
# option 1: strict fallback only when google_place_id is missing
# =========================
def find_google_place_id_by_name_location(
    name: str,
    location_hint: str | None,
    fallback_location: str = "Lebanon",
) -> str | None:
    if not name:
        return None
    name = str(name).strip()
    if not name:
        return None

    loc = (location_hint or "").strip() or fallback_location

    try:
        results = places_service.text_search(name, loc, place_type=None, max_pages=1)
    except Exception:
        return None

    if not results:
        return None

    best = None
    best_score = -1
    for r in results[:5]:
        rname = (r.get("name") or "").strip()
        if not rname:
            continue
        score = fuzz.token_set_ratio(name.lower(), rname.lower())
        if score > best_score:
            best_score = score
            best = r

    if not best or best_score < 80:
        return None

    return best.get("place_id")


def ensure_image_url_for_firestore_doc(
    collection_name: str,
    doc_id: str,
    place_profile: dict,
    location_hint: str | None = None,
) -> str | None:
    """
    1) If doc has google_place_id -> we DO NOT search.
       We just ensure the field exists and let /photo endpoint handle serving the image.
    2) If missing google_place_id -> try strict name+location match and store google_place_id if found.
    3) If cannot find -> return None.
    """
    if place_profile.get("google_place_id"):
        return None

    found_id = find_google_place_id_by_name_location(
        place_profile.get("name", ""),
        location_hint,
        fallback_location="Lebanon",
    )
    if not found_id:
        return None

    update_data = {"google_place_id": found_id}
    db.collection(collection_name).document(doc_id).set(update_data, merge=True)
    place_profile["google_place_id"] = found_id
    return None

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
                "rating": float(profile.get("rating") or 0.0),
            }

            lat = profile.get("latitude")
            lon = profile.get("longitude")
            if lat is not None and lon is not None:
                place_entry["latitude"] = lat
                place_entry["longitude"] = lon

            # OPTION 1: image URL points to OUR backend proxy endpoint
            gpid = profile.get("google_place_id")
            if gpid and PUBLIC_BASE_URL:
                place_entry["image_url"] = f"{PUBLIC_BASE_URL}/photo?place_id={gpid}"

            places.append(place_entry)

        out.append({"requirement": req, "places": places})
    return out

def match_user_to_place(profile, place_profile, typ):
    preferred_primary = profile.get("preferred_primary_features", [])
    ambiance_pref = profile.get("ambiance", [])
    preferred_secondary = profile.get("preferred_secondary_features", [])
    rating_threshold = profile.get("rating_threshold", 0.0)
    min_reviews_count = profile.get("min_reviews_count", 0)

    score = 0

    if any(f in place_profile["primary_features"] for f in preferred_primary):
        score += 5

    score += len(set(ambiance_pref) & set(place_profile["ambiance"])) * 3
    score += len(set(preferred_secondary) & set(place_profile["secondary_features"])) * 2

    if place_profile["avg_sentiment"] > 0.3:
        score += 3

    if place_profile["rating"] >= rating_threshold:
        score += 4
    if place_profile["num_reviews"] >= min_reviews_count:
        score += 2

    return score

# ========== Distance helpers (small bias + diversity) ==========
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def distance_bias_bonus(user_lat, user_lon, place_lat, place_lon) -> float:
    if user_lat is None or user_lon is None or place_lat is None or place_lon is None:
        return 0.0
    d = haversine_km(user_lat, user_lon, place_lat, place_lon)
    if d <= 2:  return 0.15
    if d <= 5:  return 0.1
    if d <= 10: return 0.06
    if d <= 20: return 0.03
    return 0.0

def diversify_by_distance(candidates: list[dict], k: int, min_spread_km: float = 2.0) -> list[dict]:
    chosen = []
    ordered = sorted(candidates, key=lambda x: x["score"], reverse=True)

    for c in ordered:
        if len(chosen) >= k:
            break
        clat = c["profile"].get("latitude")
        clon = c["profile"].get("longitude")
        if clat is None or clon is None:
            chosen.append(c)
            continue

        ok = True
        for picked in chosen:
            plat = picked["profile"].get("latitude")
            plon = picked["profile"].get("longitude")
            if plat is None or plon is None:
                continue
            if haversine_km(clat, clon, plat, plon) < min_spread_km:
                ok = False
                break
        if ok:
            chosen.append(c)

    if len(chosen) < k:
        chosen_set = set(id(x) for x in chosen)
        for c in ordered:
            if len(chosen) >= k:
                break
            if id(c) in chosen_set:
                continue
            chosen.append(c)

    return chosen

# ========== Firestore helpers ==========
def build_profile_from_firestore_doc(doc_data: dict, recommendation_type: str):
    rating = float(doc_data.get("rating", 0.0) or 0.0)
    num_reviews = int(doc_data.get("num_reviews") or len(doc_data.get("reviews") or []))

    latitude = None
    longitude = None
    loc = doc_data.get("location")
    if loc is not None:
        if hasattr(loc, "latitude") and hasattr(loc, "longitude"):
            latitude = loc.latitude
            longitude = loc.longitude
        elif isinstance(loc, dict):
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
        "google_place_id": doc_data.get("google_place_id"),
        "raw": doc_data,
    }

def upsert_place_into_firestore(collection_name: str, place_details: dict, recommendation_type: str):
    place_id = place_details.get("place_id")
    name = place_details.get("name", "Unknown")
    address = place_details.get("address", "N/A")

    doc_id = place_id if place_id else f"{name}_{address}".replace("/", "_")[:150]

    update_data = {
        "name": name,
        "address": address,
        "rating": float(place_details.get("rating") or 0.0),
        "num_reviews": int(place_details.get("user_ratings_total") or 0),
        "reviews": place_details.get("reviews") or [],
        "types": place_details.get("types") or [],
        "google_place_id": place_id,
    }

    lat = place_details.get("latitude")
    lon = place_details.get("longitude")
    if lat is not None and lon is not None:
        update_data["location"] = GeoPoint(float(lat), float(lon))

    profile = extract_place_profile(place_details, recommendation_type)
    update_data.update({
        "avg_sentiment": profile["avg_sentiment"],
        "primary_features": profile["primary_features"],
        "secondary_features": profile["secondary_features"],
        "ambiance": profile["ambiance"],
    })

    db.collection(collection_name).document(doc_id).set(update_data, merge=True)
    return doc_id, None
    
def _get_lat_lon_from_profile(place_profile: dict) -> tuple[float | None, float | None]:
    lat = place_profile.get("latitude")
    lon = place_profile.get("longitude")
    if lat is None or lon is None:
        return None, None
    try:
        return float(lat), float(lon)
    except Exception:
        return None, None


def within_radius_km(
    user_lat: float,
    user_lon: float,
    place_lat: float,
    place_lon: float,
    radius_km: float
) -> bool:
    return haversine_km(user_lat, user_lon, place_lat, place_lon) <= float(radius_km)


'''def pick_random_zone_center_from_candidates(
    candidates: list[dict],
    seed: int | None = None
) -> tuple[float, float] | None:
    """
    candidates: list of dicts that contain profile.latitude/profile.longitude (or already in profile)
    Returns (lat, lon) of a randomly chosen candidate as the zone center.
    """
    rng = random.Random(seed)
    good = []
    for c in candidates:
        p = c.get("profile") or {}
        lat, lon = _get_lat_lon_from_profile(p)
        if lat is not None and lon is not None:
            good.append((lat, lon))
    if not good:
        return None
    return rng.choice(good) '''
    
def pick_random_zone_center_diverse(
    scored: list[dict],
    seed: int | None,
    centers_k: int = 25,
    min_spread_km: float = 8.0,
) -> tuple[float, float] | None:
    """
    Picks a random zone center from a set of geographically spread-out candidates.
    This avoids always landing in Beirut if your top-scored pool is Beirut-heavy.
    """
    rng = random.Random(seed)

    # Build candidate list of items that have lat/lon
    candidates = []
    for item in scored:
        p = item.get("profile") or {}
        lat, lon = _get_lat_lon_from_profile(p)
        if lat is not None and lon is not None:
            candidates.append((lat, lon))

    if not candidates:
        return None

    # Shuffle for randomness, then greedily keep spread-out centers
    rng.shuffle(candidates)

    centers: list[tuple[float, float]] = []
    for lat, lon in candidates:
        ok = True
        for clat, clon in centers:
            if haversine_km(lat, lon, clat, clon) < min_spread_km:
                ok = False
                break
        if ok:
            centers.append((lat, lon))
        if len(centers) >= centers_k:
            break

    if not centers:
        return None

    return rng.choice(centers)

def rank_places_from_firestore(
    collection_name: str,
    user_profile: dict,
    recommendation_type: str,
    top_n: int,
    query: str = "",
    user_latitude: float | None = None,
    user_longitude: float | None = None,
    force_location: bool = False,
    location_hint: str | None = None,

    zone_mode: bool = False,
    zone_radius_km: float = 8.0,
    zone_center: tuple[float, float] | None = None,
    zone_seed: int | None = None,
    zone_pool_size: int = 250,

    exclude_place_ids: list[str] | None = None,   # NEW
    exclude_doc_ids: list[str] | None = None,     # NEW (fallback)
):
    exclude_place_ids = set((exclude_place_ids or []))
    exclude_doc_ids = set((exclude_doc_ids or []))

    docs = db.collection(collection_name).stream()
    q_tokens = preprocess_text(query)
    scored = []

    loc_lower = (location_hint or "").strip().lower()

    for snap in docs:
        if snap.id in exclude_doc_ids:
            continue

        data = snap.to_dict() or {}
        if "primary_features" not in data or "avg_sentiment" not in data:
            continue

        place_profile = build_profile_from_firestore_doc(data, recommendation_type)

        # Skip if google_place_id already shown
        gpid = place_profile.get("google_place_id")
        if gpid and gpid in exclude_place_ids:
            continue

        if force_location and loc_lower:
            addr = (data.get("address") or "").lower()
            name = (data.get("name") or "").lower()
            if loc_lower not in addr and loc_lower not in name:
                continue

        s = match_user_to_place(user_profile, place_profile, recommendation_type)

        if q_tokens:
            combined_features = place_profile["primary_features"] + place_profile["secondary_features"]
            if any(qt in combined_features for qt in q_tokens):
                s += 4

        if not force_location:
            s += distance_bias_bonus(
                user_latitude, user_longitude,
                place_profile.get("latitude"), place_profile.get("longitude"),
            )

        scored.append({
            "name": place_profile["name"],
            "score": float(s),
            "profile": place_profile,
            "doc_id": snap.id,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    # -------------------------
    # ZONE MODE (random pin + radius filter)
    # -------------------------
    if zone_mode and scored:
        # Choose center
        if zone_center is None:
            pool = scored[:max(zone_pool_size, top_n * 10)]
            zone_used = pick_random_zone_center_diverse(pool, seed=zone_seed,centers_k=25,min_spread_km=8.0,)
        else:
            zone_used = zone_center

        if zone_used is not None:
            zlat, zlon = zone_used

            zone_scored = []
            for item in scored:
                p = item.get("profile") or {}
                plat, plon = _get_lat_lon_from_profile(p)
                if plat is None or plon is None:
                    continue
                if within_radius_km(zlat, zlon, plat, plon, zone_radius_km):
                    zone_scored.append(item)

            # widen once if too small
            if len(zone_scored) < max(3, top_n):
                widened = float(zone_radius_km) * 1.8
                zone_scored2 = []
                for item in scored:
                    p = item.get("profile") or {}
                    plat, plon = _get_lat_lon_from_profile(p)
                    if plat is None or plon is None:
                        continue
                    if within_radius_km(zlat, zlon, plat, plon, widened):
                        zone_scored2.append(item)
                if len(zone_scored2) >= len(zone_scored):
                    zone_scored = zone_scored2

            # fallback if still tiny
            if len(zone_scored) >= max(3, top_n):
                scored = zone_scored

    # -------------------------
    # Final selection
    # -------------------------
    diversified = diversify_by_distance(scored, k=top_n, min_spread_km=2.0)[:top_n]

    for item in diversified:
        ensure_image_url_for_firestore_doc(
            collection_name,
            item["doc_id"],
            item["profile"],
            location_hint=location_hint,
        )

    return diversified

# ========== Existing Firestore trip (fallback) ==========
def firestore_trip_generate(
    user_food: dict,
    user_act: dict,
    num_restaurants: int = 1,
    num_activities: int = 2,
    location: str = "Beirut, Lebanon",
    user_latitude: float | None = None,
    user_longitude: float | None = None,
):
    plan = []

    food_kw = " ".join(user_food.get("preferred_primary_features", [])[:2]) or "restaurant"
    act_kw = " ".join(user_act.get("preferred_primary_features", [])[:3]) or "activity"

    # Use milliseconds so back-to-back requests are different
    seed_base = int(time.time() * 1000)

    chosen_place_ids: set[str] = set()
    chosen_doc_ids: set[str] = set()

    # -------- Restaurants: each item gets its own random zone --------
    restaurants: list[dict] = []
    for i in range(int(num_restaurants)):
        recs = rank_places_from_firestore(
            collection_name="RestaurantsFinal",
            user_profile=user_food,
            recommendation_type="food",
            top_n=1,
            query=food_kw,
            user_latitude=user_latitude,
            user_longitude=user_longitude,
            force_location=False,
            location_hint=location,

            zone_mode=True,
            zone_radius_km=6.0,
            zone_seed=seed_base + 1000 + i,  # DIFFERENT SEED PER ITEM

            exclude_place_ids=list(chosen_place_ids),
            exclude_doc_ids=list(chosen_doc_ids),
        )

        if not recs:
            continue

        pick = recs[0]
        restaurants.append(pick)

        gpid = (pick.get("profile") or {}).get("google_place_id")
        if gpid:
            chosen_place_ids.add(gpid)
        if pick.get("doc_id"):
            chosen_doc_ids.add(pick["doc_id"])

    plan.append({
        "requirement": {"type": "restaurant", "count": num_restaurants, "keywords": [food_kw], "location_hint": None},
        "recommendations": restaurants,
    })

    # -------- Activities: each item gets its own random zone --------
    activities: list[dict] = []
    for i in range(int(num_activities)):
        recs = rank_places_from_firestore(
            collection_name="ActivitiesFinal",
            user_profile=user_act,
            recommendation_type="activity",
            top_n=1,
            query=act_kw,
            user_latitude=user_latitude,
            user_longitude=user_longitude,
            force_location=False,
            location_hint=location,

            zone_mode=True,
            zone_radius_km=10.0,
            zone_seed=seed_base + 2000 + i,  # DIFFERENT SEED PER ITEM

            exclude_place_ids=list(chosen_place_ids),
            exclude_doc_ids=list(chosen_doc_ids),
        )

        if not recs:
            continue

        pick = recs[0]
        activities.append(pick)

        gpid = (pick.get("profile") or {}).get("google_place_id")
        if gpid:
            chosen_place_ids.add(gpid)
        if pick.get("doc_id"):
            chosen_doc_ids.add(pick["doc_id"])

    plan.append({
        "requirement": {"type": "activity", "count": num_activities, "keywords": [act_kw], "location_hint": None},
        "recommendations": activities,
    })

    return {"plan": plan}


# ========== Query-driven Firestore trip ==========
def firestore_trip_generate_from_query(
    query: str,
    user_food: dict,
    user_act: dict,
    user_latitude: float | None = None,
    user_longitude: float | None = None,
):
    """
    Query-driven trip generator.
    - Uses OpenAI to extract requirements
    - Searches **Google Places API only**, ignores Firestore.
    """
    extraction = extract_requirements_with_openai(query)
    requirements = extraction.get("requirements", []) or []

    if not requirements:
        # Fallback to Firestore if extraction fails
        return {"extracted": extraction, "plan": firestore_trip_generate(
            user_food, user_act,
            num_restaurants=1, num_activities=2,
            location="Beirut, Lebanon",
            user_latitude=user_latitude,
            user_longitude=user_longitude,
        )["plan"]}

    gps = GooglePlacesService()
    plan = []

    for req_item in requirements:
        typ = req_item.get("type")
        count = int(req_item.get("count", 1))
        keywords = req_item.get("keywords", []) or []
        location_hint = req_item.get("location_hint") or "Lebanon"

        query_str = " ".join(keywords).strip() or typ

        # --- GOOGLE PLACES ONLY ---
        results = gps.text_search(query_str, location_hint, max_pages=1)[:count]

        recommendations = []
        for r in results:
            profile = extract_place_profile(r, "food" if typ == "restaurant" else "activity")
            score = match_user_to_place(
                user_food if typ=="restaurant" else user_act,
                profile,
                typ
            )
            recommendations.append({
                "name": r.get("name"),
                "score": round(float(score), 2),
                "profile": profile
            })

        plan.append({
            "requirement": {
                "type": typ,
                "count": count,
                "keywords": keywords,
                "location_hint": location_hint,
                "time_hint": req_item.get("time_hint")
            },
            "recommendations": recommendations
        })

    return {"extracted": extraction, "plan": plan}

# ========== Firestore enrichment (unchanged) ==========
def build_place_dict_from_firestore_doc(doc_data: dict) -> dict:
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
    print(f"=== Enriching collection: {collection_name} (type={recommendation_type}) ===")

    docs = db.collection(collection_name).stream()
    processed = 0
    skipped_no_reviews = 0

    for snap in docs:
        doc_id = snap.id
        data = snap.to_dict() or {}

        reviews = data.get("reviews") or []
        if not reviews:
            skipped_no_reviews += 1
            continue

        if "avg_sentiment" in data and "primary_features" in data:
            continue

        place_dict = build_place_dict_from_firestore_doc(data)
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
        }

        db.collection(collection_name).document(doc_id).set(update_data, merge=True)
        processed += 1

    print("\n=== Summary for", collection_name, "===")
    print("Processed docs:", processed)
    print("Skipped (no reviews):", skipped_no_reviews)
    print("====================================\n")
