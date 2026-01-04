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

from nltk.corpus import wordnet as wn

try:
    wn.ensure_loaded()
except Exception:
    _ = wn.synsets("dog")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

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

def build_places_photo_url(photo_reference: str, maxwidth: int = 1200) -> str | None:
    if not photo_reference or not GOOGLE_API_KEY:
        return None
    return f"{BASE_URL}/photo?maxwidth={maxwidth}&photo_reference={photo_reference}&key={GOOGLE_API_KEY}"

class GooglePlacesService:
    def __init__(self, api_key=GOOGLE_API_KEY):
        self.api_key = api_key
        self.session = requests.Session()

    def _details(self, place_id):
        # NEW: include place_id, photos, geometry for lat/lon
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
            url = f"{BASE_URL}/textsearch/json?query={requests.utils.quote(query + ' in ' + location_str)}&key={self.api_key}"
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

def find_place_id_by_name_address(name: str, address: str, fallback_location: str = "Lebanon") -> str | None:
    """
    If a Firestore doc doesn't have place_id, try to find it via Google Places Text Search.
    Keeps it lightweight: 1 page.
    """
    if not name:
        return None
    q = name
    if address:
        q = f"{name} {address}"
    try:
        results = places_service.text_search(q, fallback_location, place_type=None, max_pages=1)
        if results:
            return results[0].get("place_id")
    except Exception:
        pass
    return None


def ensure_image_url_for_firestore_doc(
    collection_name: str,
    doc_id: str,
    place_profile: dict,
) -> str | None:
    # Already has one
    if place_profile.get("image_url"):
        return place_profile["image_url"]

    place_id = place_profile.get("place_id")

    # If no place_id, try to recover it (expensive + possibly wrong)
    if not place_id:
        name = place_profile.get("name", "")
        address = place_profile.get("address", "")
        if name and address:  # only do it when we have both
            place_id = find_place_id_by_name_address(name, address, fallback_location="Lebanon")
            if place_id:
                place_profile["place_id"] = place_id

    if not place_id:
        return None

    details = places_service._details(place_id)
    if not details:
        return None

    image_url = build_places_photo_url(details.get("photo_reference"))

    update_data = {}

    # Save place_id if missing
    if not place_profile.get("place_id") and place_id:
        update_data["place_id"] = place_id

    # Save image_url if found
    if image_url:
        update_data["image_url"] = image_url
        place_profile["image_url"] = image_url

    # Patch missing lat/lon on the profile + Firestore if profile has none
    lat = details.get("latitude")
    lon = details.get("longitude")
    if lat is not None and lon is not None:
        if place_profile.get("latitude") is None or place_profile.get("longitude") is None:
            place_profile["latitude"] = lat
            place_profile["longitude"] = lon
            update_data["location"] = GeoPoint(float(lat), float(lon))

    if update_data:
        db.collection(collection_name).document(doc_id).set(update_data, merge=True)

    return place_profile.get("image_url")

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

            # NEW: image_url support
            img = profile.get("image_url")
            if img:
                place_entry["image_url"] = img

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
    # Gentle bias only. Max +1.5
    if user_lat is None or user_lon is None or place_lat is None or place_lon is None:
        return 0.0
    d = haversine_km(user_lat, user_lon, place_lat, place_lon)
    if d <= 2:  return 1.5
    if d <= 5:  return 1.0
    if d <= 10: return 0.6
    if d <= 20: return 0.3
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
        "image_url": doc_data.get("image_url"),   # NEW
        "place_id": doc_data.get("place_id"),     # NEW
        "raw": doc_data,
    }

def upsert_place_into_firestore(collection_name: str, place_details: dict, recommendation_type: str):
    """
    Upsert place + image_url + location geopoint + enrichment fields so it becomes rankable from Firestore next time.
    """
    place_id = place_details.get("place_id")
    name = place_details.get("name", "Unknown")
    address = place_details.get("address", "N/A")

    doc_id = place_id if place_id else f"{name}_{address}".replace("/", "_")[:150]

    image_url = build_places_photo_url(place_details.get("photo_reference"))

    update_data = {
        "name": name,
        "address": address,
        "rating": float(place_details.get("rating") or 0.0),
        "num_reviews": int(place_details.get("user_ratings_total") or 0),
        "reviews": place_details.get("reviews") or [],
        "types": place_details.get("types") or [],
        "place_id": place_id,
    }

    lat = place_details.get("latitude")
    lon = place_details.get("longitude")
    if lat is not None and lon is not None:
        update_data["location"] = GeoPoint(float(lat), float(lon))

    if image_url:
        update_data["image_url"] = image_url

    # Enrich immediately using your existing extractor
    profile = extract_place_profile(place_details, recommendation_type)
    update_data.update({
        "avg_sentiment": profile["avg_sentiment"],
        "primary_features": profile["primary_features"],
        "secondary_features": profile["secondary_features"],
        "ambiance": profile["ambiance"],
    })

    db.collection(collection_name).document(doc_id).set(update_data, merge=True)
    return doc_id, image_url

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
):
    """
    Firestore scoring with:
    - your profile score
    - optional query boost
    - SMALL distance bias (unless force_location)
    - optional lightweight location_hint filter if force_location=True
    - diversity rerank
    - NEW: if picked docs have no image_url, fetch via Google Places + update Firestore + return it
    """
    docs = db.collection(collection_name).stream()
    q_tokens = preprocess_text(query)
    scored = []

    loc_lower = (location_hint or "").strip().lower()

    for snap in docs:
        data = snap.to_dict() or {}

        # only consider enriched docs
        if "primary_features" not in data or "avg_sentiment" not in data:
            continue

        # If query forces a location, do a lightweight filter (no schema assumptions)
        if force_location and loc_lower:
            addr = (data.get("address") or "").lower()
            name = (data.get("name") or "").lower()
            if loc_lower not in addr and loc_lower not in name:
                continue

        place_profile = build_profile_from_firestore_doc(data, recommendation_type)
        s = match_user_to_place(user_profile, place_profile, recommendation_type)

        # query boost
        if q_tokens:
            combined_features = place_profile["primary_features"] + place_profile["secondary_features"]
            if any(qt in combined_features for qt in q_tokens):
                s += 4

        # small distance bias unless forced location
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

    # Sort then diversify
    scored.sort(key=lambda x: x["score"], reverse=True)
    diversified = diversify_by_distance(scored, k=top_n, min_spread_km=2.0)
    diversified = diversified[:top_n]

    # NEW: lazy-enrich image_url only for the picked results
    for item in diversified:
        prof = item["profile"]
        doc_id = item["doc_id"]
        # If missing image_url, fetch + store + return
        ensure_image_url_for_firestore_doc(collection_name, doc_id, prof)

    return diversified

# ========== Existing endpoints support (kept) ==========
def auto_trip_generate(user_food, user_act, location="Beirut, Lebanon"):
    # left unchanged; depends on rank_places(...) which isn't in your paste
    # You can keep it or remove it if you don't use /auto-trip
    plan = []
    food_kw = " ".join(user_food.get("preferred_primary_features", [])[:2]) or "restaurant"
    raw_food = places_service.text_search(food_kw, location, place_type="restaurant")
    ranked_food = []  # optional: you can remove auto-trip if unused
    plan.append({
        "requirement": {"type": "restaurant", "count": 1, "keywords": [food_kw], "location_hint": location},
        "recommendations": ranked_food
    })

    act_kw = " ".join(user_act.get("preferred_primary_features", [])[:3]) or "attraction"
    raw_act = places_service.text_search(act_kw, location, place_type=None)
    ranked_act = []
    plan.append({
        "requirement": {"type": "activity", "count": 2, "keywords": [act_kw], "location_hint": location},
        "recommendations": ranked_act
    })
    return {"plan": plan}

# ========== Query-driven Firestore trip (NEW core) ==========
def firestore_trip_generate_from_query(
    query: str,
    user_food: dict,
    user_act: dict,
    user_latitude: float | None = None,
    user_longitude: float | None = None,
):
    extraction = extract_requirements_with_openai(query)
    requirements = extraction.get("requirements", []) or []

    # If extraction fails, fall back to old default behavior
    if not requirements:
        return {"extracted": extraction, "plan": firestore_trip_generate(
            user_food, user_act,
            num_restaurants=1, num_activities=2,
            location="Beirut, Lebanon",
            user_latitude=user_latitude,
            user_longitude=user_longitude,
        )["plan"]}

    plan = []

    for req in requirements:
        typ = req.get("type")
        count = int(req.get("count", 1))
        keywords = req.get("keywords", []) or []
        kw = " ".join(keywords).strip() or (typ or "")

        location_hint = req.get("location_hint")
        force_location = bool(location_hint and str(location_hint).strip())

        if typ == "restaurant":
            # 1) Firestore first
            recs = rank_places_from_firestore(
                collection_name="RestaurantsFinal",
                user_profile=user_food,
                recommendation_type="food",
                top_n=count,
                query=kw,
                user_latitude=user_latitude,
                user_longitude=user_longitude,
                force_location=force_location,
                location_hint=location_hint,
            )

            missing = count - len(recs)

            # 2) Google fallback if missing
            if missing > 0:
                # If force_location, search in that location; else use a very general Lebanon bias.
                # We don't have request.location anymore; we keep the search broad.
                search_loc = location_hint if force_location else "Lebanon"
                raw = places_service.text_search(kw, search_loc, place_type="restaurant", max_pages=1)

                google_candidates = []
                for p in raw:
                    # profile from Places reviews
                    place_profile = extract_place_profile(p, "food")
                    s = match_user_to_place(user_food, place_profile, "food")

                    # small distance bias unless forced location
                    if not force_location:
                        s += distance_bias_bonus(
                            user_latitude, user_longitude,
                            p.get("latitude"), p.get("longitude"),
                        )

                    # upsert + get image url
                    _, image_url = upsert_place_into_firestore("RestaurantsFinal", p, "food")

                    # include fields in response profile
                    place_profile["image_url"] = image_url
                    place_profile["latitude"] = p.get("latitude")
                    place_profile["longitude"] = p.get("longitude")
                    place_profile["place_id"] = p.get("place_id")

                    google_candidates.append({
                        "name": place_profile["name"],
                        "score": float(s),
                        "profile": place_profile,
                        "doc_id": p.get("place_id") or "",
                    })

                # sort + diversify then take missing
                google_candidates.sort(key=lambda x: x["score"], reverse=True)
                google_picks = diversify_by_distance(google_candidates, k=missing, min_spread_km=2.0)
                recs.extend(google_picks)

            plan.append({
                "requirement": {
                    "type": "restaurant",
                    "count": count,
                    "keywords": keywords,
                    "location_hint": location_hint,
                    "time_hint": req.get("time_hint"),
                },
                "recommendations": recs[:count],
            })

        else:
            # Activities
            recs = rank_places_from_firestore(
                collection_name="ActivitiesFinal",
                user_profile=user_act,
                recommendation_type="activity",
                top_n=count,
                query=kw,
                user_latitude=user_latitude,
                user_longitude=user_longitude,
                force_location=force_location,
                location_hint=location_hint,
            )

            missing = count - len(recs)

            if missing > 0:
                search_loc = location_hint if force_location else "Lebanon"
                raw = places_service.text_search(kw, search_loc, place_type=None, max_pages=1)

                google_candidates = []
                for p in raw:
                    place_profile = extract_place_profile(p, "activity")
                    s = match_user_to_place(user_act, place_profile, "activity")

                    if not force_location:
                        s += distance_bias_bonus(
                            user_latitude, user_longitude,
                            p.get("latitude"), p.get("longitude"),
                        )

                    _, image_url = upsert_place_into_firestore("ActivitiesFinal", p, "activity")

                    place_profile["image_url"] = image_url
                    place_profile["latitude"] = p.get("latitude")
                    place_profile["longitude"] = p.get("longitude")
                    place_profile["place_id"] = p.get("place_id")

                    google_candidates.append({
                        "name": place_profile["name"],
                        "score": float(s),
                        "profile": place_profile,
                        "doc_id": p.get("place_id") or "",
                    })

                google_candidates.sort(key=lambda x: x["score"], reverse=True)
                google_picks = diversify_by_distance(google_candidates, k=missing, min_spread_km=2.0)
                recs.extend(google_picks)

            plan.append({
                "requirement": {
                    "type": "activity",
                    "count": count,
                    "keywords": keywords,
                    "location_hint": location_hint,
                    "time_hint": req.get("time_hint"),
                },
                "recommendations": recs[:count],
            })

    return {"extracted": extraction, "plan": plan}

# ========== Existing Firestore trip (kept as fallback) ==========
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
    top_restaurants = rank_places_from_firestore(
        collection_name="RestaurantsFinal",
        user_profile=user_food,
        recommendation_type="food",
        top_n=num_restaurants,
        query=food_kw,
        user_latitude=user_latitude,
        user_longitude=user_longitude,
        force_location=False,
    )
    plan.append({
        "requirement": {
            "type": "restaurant",
            "count": num_restaurants,
            "keywords": [food_kw],
            "location_hint": None,
        },
        "recommendations": top_restaurants,
    })

    act_kw = " ".join(user_act.get("preferred_primary_features", [])[:3]) or "activity"
    top_activities = rank_places_from_firestore(
        collection_name="ActivitiesFinal",
        user_profile=user_act,
        recommendation_type="activity",
        top_n=num_activities,
        query=act_kw,
        user_latitude=user_latitude,
        user_longitude=user_longitude,
        force_location=False,
    )
    plan.append({
        "requirement": {
            "type": "activity",
            "count": num_activities,
            "keywords": [act_kw],
            "location_hint": None,
        },
        "recommendations": top_activities,
    })

    return {"plan": plan}

# ========== Firestore enrichment (unchanged, from your version) ==========
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
