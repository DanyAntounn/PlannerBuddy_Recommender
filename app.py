# app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from typing import Optional, Union, List
from chatbot import router as chatbot_router
from recommender import haversine_km
from recommender import db

import os
import requests

from recommender import (
    build_flutter_payload,
    enrich_collection_with_profiles_from_firestore,
    firestore_trip_generate,                # fallback (no query)
    firestore_trip_generate_from_query,     # query-driven
    rank_places_from_firestore,
    match_user_to_user,
)

app = FastAPI(title="Trip Planner API")
app.include_router(chatbot_router)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
BASE_URL = "https://maps.googleapis.com/maps/api/place"
_session = requests.Session()


class UserProfile(BaseModel):
    preferred_primary_features: list[str]
    ambiance: list[str]
    preferred_secondary_features: list[str]
    rating_threshold: float
    min_reviews_count: int


class LocationPayload(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None


class FirestoreTripRequest(BaseModel):
    query: Optional[str] = None

    user_food: UserProfile
    user_act: UserProfile

    personality_profile: Optional[dict] = None  # <-- ADD THIS

    num_restaurants: int = 1
    num_activities: int = 2

    location: Optional[Union[str, LocationPayload]] = None

    latitude: Optional[float] = None
    longitude: Optional[float] = None


class MapFilterRequest(BaseModel):
    user_food: UserProfile
    user_act: UserProfile

    # optional: user location for distance bias
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    # optional: helps some “force_location” filters if you ever enable them
    location_hint: Optional[str] = "Lebanon"

    # how many to return
    num_restaurants: int = 10
    num_activities: int = 10

    # optional: if your UI has a free-text “intent” field (ex: “sushi”, “hiking”)
    restaurant_intent: Optional[str] = ""
    activity_intent: Optional[str] = ""

class MeetupProfile(BaseModel):
    vibe_preferences: List[str] = []
    noise_tolerance: float = 0.5
    cultural_interest: float = 0.5
    budget_sensitivity: float = 0.5
    social_energy: float = 0.5
    group_style: Optional[str] = None  # "solo" | "group" | etc.

class MeetupMatchesRequest(BaseModel):
    user_id: str
    profile: MeetupProfile
    latitude: float
    longitude: float
    radius_km: float = 5.0
    limit: int = 10

def _get_photo_reference(place_id: str) -> str | None:
    if not GOOGLE_API_KEY:
        return None

    url = (
        f"{BASE_URL}/details/json"
        f"?place_id={place_id}"
        f"&fields=photos"
        f"&key={GOOGLE_API_KEY}"
    )
    try:
        r = _session.get(url, timeout=20)
        data = r.json()
    except Exception:
        return None

    if data.get("status") != "OK":
        return None

    photos = (data.get("result") or {}).get("photos") or []
    if not photos:
        return None

    return photos[0].get("photo_reference")


@app.get("/photo")
def photo(place_id: str, maxwidth: int = 1200):
    """
    Backend image proxy.
    - Client calls /photo?place_id=ChIJ...
    - We fetch the Google Places photo using GOOGLE_API_KEY (kept secret on server)
    - We stream the image back (Flutter can Image.network this endpoint)
    """
    place_id = (place_id or "").strip()
    if not place_id:
        raise HTTPException(status_code=400, detail="place_id is required")

    photo_ref = _get_photo_reference(place_id)
    if not photo_ref:
        return Response(status_code=204)

    photo_url = (
        f"{BASE_URL}/photo"
        f"?maxwidth={maxwidth}"
        f"&photo_reference={photo_ref}"
        f"&key={GOOGLE_API_KEY}"
    )

    try:
        upstream = _session.get(photo_url, stream=True, timeout=30)
    except Exception:
        return Response(status_code=204)

    if upstream.status_code != 200:
        return Response(status_code=204)

    content_type = upstream.headers.get("Content-Type", "image/jpeg")

    return StreamingResponse(
        upstream.raw,
        media_type=content_type,
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/enrich_firestore")
def enrich_firestore():
    enrich_collection_with_profiles_from_firestore("RestaurantsFinal", "food")
    enrich_collection_with_profiles_from_firestore("ActivitiesFinal", "activity")
    return {"status": "OK"}


@app.post("/firestore-trip")
def firestore_trip(req: FirestoreTripRequest):
    user_lat = req.latitude
    user_lon = req.longitude

    if (user_lat is None or user_lon is None) and isinstance(req.location, LocationPayload):
        user_lat = req.location.lat
        user_lon = req.location.lon

    if req.query and req.query.strip():
        res = firestore_trip_generate_from_query(
            query=req.query,
            user_food=req.user_food.dict(),
            user_act=req.user_act.dict(),
            personality_profile=req.personality_profile,  # <-- ADD THIS
            user_latitude=user_lat,
            user_longitude=user_lon,
        )
        flutter_payload = build_flutter_payload(res["plan"])
        return {"extracted": res["extracted"], "plan": flutter_payload}

    num_restaurants = req.num_restaurants if req.num_restaurants > 0 else 1
    num_activities = req.num_activities if req.num_activities > 0 else 2

    res = firestore_trip_generate(
        req.user_food.dict(),
        req.user_act.dict(),
        personality_profile=req.personality_profile, 
        num_restaurants=num_restaurants,
        num_activities=num_activities,
        location="Lebanon",
        user_latitude=user_lat,
        user_longitude=user_lon,
    )
    flutter_payload = build_flutter_payload(res["plan"])
    return {"plan": flutter_payload}

@app.post("/map-filter")
def map_filter(req: MapFilterRequest):
    num_restaurants = max(1, min(int(req.num_restaurants or 10), 20))
    num_activities = max(1, min(int(req.num_activities or 10), 20))

    user_lat = req.latitude
    user_lon = req.longitude
    location_hint = (req.location_hint or "Lebanon").strip() or "Lebanon"

    restaurants = rank_places_from_firestore(
        collection_name="RestaurantsFinal",
        user_profile=req.user_food.dict(),
        recommendation_type="food",
        top_n=num_restaurants,
        query=(req.restaurant_intent or ""),
        user_latitude=user_lat,
        user_longitude=user_lon,
        force_location=False,
        location_hint=location_hint,
    )

    activities = rank_places_from_firestore(
        collection_name="ActivitiesFinal",
        user_profile=req.user_act.dict(),
        recommendation_type="activity",
        top_n=num_activities,
        query=(req.activity_intent or ""),
        user_latitude=user_lat,
        user_longitude=user_lon,
        force_location=False,
        location_hint=location_hint,
    )

@app.post("/meetup-matches")
def meetup_matches(req: MeetupMatchesRequest):
    user_lat = float(req.latitude)
    user_lon = float(req.longitude)
    radius_km = float(req.radius_km or 5.0)
    limit = max(1, min(int(req.limit or 10), 50))

    me_profile = req.profile.dict()
    me_id = (req.user_id or "").strip()
    if not me_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    candidates = []
    docs = db.collection("PersonalityProfile").stream()

    for snap in docs:
        data = snap.to_dict() or {}

        other_id = (data.get("userId") or data.get("user_id") or snap.id)
        if str(other_id) == me_id:
            continue

        # location extraction: supports latitude/longitude fields OR a GeoPoint in "location"
        lat = data.get("latitude")
        lon = data.get("longitude")

        loc = data.get("location")
        if (lat is None or lon is None) and loc is not None:
            if hasattr(loc, "latitude") and hasattr(loc, "longitude"):
                lat = loc.latitude
                lon = loc.longitude
            elif isinstance(loc, dict):
                lat = loc.get("latitude")
                lon = loc.get("longitude")

        if lat is None or lon is None:
            continue

        try:
            lat = float(lat)
            lon = float(lon)
        except Exception:
            continue

        d = haversine_km(user_lat, user_lon, lat, lon)
        if d > radius_km:
            continue

        other_profile = {
            "vibe_preferences": data.get("vibe_preferences", []) or [],
            "noise_tolerance": data.get("noise_tolerance", 0.5),
            "cultural_interest": data.get("cultural_interest", 0.5),
            "budget_sensitivity": data.get("budget_sensitivity", 0.5),
            "social_energy": data.get("social_energy", 0.5),
            "group_style": data.get("group_style"),
        }

        s = match_user_to_user(me_profile, other_profile)

        candidates.append({
            "user_id": str(other_id),
            "score": float(s),
            "distance_km": round(float(d), 2),
            "latitude": lat,
            "longitude": lon,
            "display_name": data.get("display_name") or data.get("name"),
            "vibe_preferences": other_profile["vibe_preferences"],
            "group_style": other_profile["group_style"],
        })

    candidates.sort(key=lambda x: (-x["score"], x["distance_km"]))

    return {
        "radius_km": radius_km,
        "count": len(candidates[:limit]),
        "matches": candidates[:limit],
    }
    
    def match_user_to_user(me: dict, other: dict) -> float:
    score = 0.0

    my_vibes = set(me.get("vibe_preferences", []) or [])
    their_vibes = set(other.get("vibe_preferences", []) or [])
    score += len(my_vibes & their_vibes) * 4

    def sim(a, b, weight):
        if a is None or b is None:
            return 0.0
        return (1.0 - abs(float(a) - float(b))) * float(weight)

    score += sim(me.get("noise_tolerance"), other.get("noise_tolerance"), 3)
    score += sim(me.get("social_energy"), other.get("social_energy"), 3)
    score += sim(me.get("cultural_interest"), other.get("cultural_interest"), 2)

    if me.get("group_style") and me.get("group_style") == other.get("group_style"):
        score += 2

    if (
        me.get("budget_sensitivity") is not None
        and other.get("budget_sensitivity") is not None
        and abs(float(me["budget_sensitivity"]) - float(other["budget_sensitivity"])) > 0.5
    ):
        score -= 2

    return round(float(score), 2)

    def to_card(item: dict, typ: str) -> dict:
        p = item.get("profile") or {}
        gpid = p.get("google_place_id")

        card = {
            "type": typ,  # "restaurant" or "activity"
            "name": item.get("name"),
            "score": float(item.get("score") or 0.0),
            "rating": float(p.get("rating") or 0.0),
            "num_reviews": int(p.get("num_reviews") or 0),
            "address": p.get("address"),
            "place_id": gpid,
            "latitude": p.get("latitude"),
            "longitude": p.get("longitude"),
        }

        if gpid and os.getenv("PUBLIC_BASE_URL", "").strip():
            base = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
            card["image_url"] = f"{base}/photo?place_id={gpid}"

        return card

    return {
        "restaurants": [to_card(x, "restaurant") for x in restaurants],
        "activities": [to_card(x, "activity") for x in activities],
    }


