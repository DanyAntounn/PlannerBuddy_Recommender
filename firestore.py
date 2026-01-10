# repositories/firestore.py
import os, json
from google.cloud import firestore
from google.oauth2 import service_account
from google.cloud.firestore_v1 import GeoPoint
from nlp import extract_place_profile
from scoring import match_user_to_place, distance_bias_bonus, diversify_by_distance

def get_firestore_client():
    secret_path = "/etc/secrets/service-account.json"
    if os.path.exists(secret_path):
        with open(secret_path) as f:
            info = json.load(f)
        creds = service_account.Credentials.from_service_account_info(info)
        return firestore.Client(credentials=creds, project=info["project_id"])
    return firestore.Client()

db = get_firestore_client()

def build_profile_from_firestore_doc(doc_data, recommendation_type):
    loc = doc_data.get("location")
    latitude = getattr(loc, "latitude", None)
    longitude = getattr(loc, "longitude", None)

    return {
        "name": doc_data.get("name", "Unknown"),
        "avg_sentiment": doc_data.get("avg_sentiment", 0.0),
        "primary_features": doc_data.get("primary_features", []),
        "secondary_features": doc_data.get("secondary_features", []),
        "ambiance": doc_data.get("ambiance", []),
        "rating": doc_data.get("rating", 0.0),
        "num_reviews": doc_data.get("num_reviews", 0),
        "address": doc_data.get("address", "N/A"),
        "types": doc_data.get("types", []),
        "latitude": latitude,
        "longitude": longitude,
        "google_place_id": doc_data.get("google_place_id"),
        "raw": doc_data,
    }

def rank_places_from_firestore(
    collection_name,
    user_profile,
    recommendation_type,
    top_n,
    query="",
    user_latitude=None,
    user_longitude=None,
    force_location=False,
    location_hint=None,
):
    docs = db.collection(collection_name).stream()
    scored = []

    for snap in docs:
        data = snap.to_dict() or {}
        if "primary_features" not in data:
            continue

        place_profile = build_profile_from_firestore_doc(data, recommendation_type)
        s = match_user_to_place(user_profile, place_profile, recommendation_type)
        s += distance_bias_bonus(user_latitude, user_longitude,
                                 place_profile.get("latitude"),
                                 place_profile.get("longitude"))

        scored.append({
            "name": place_profile["name"],
            "score": float(s),
            "profile": place_profile,
            "doc_id": snap.id,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return diversify_by_distance(scored, top_n)
