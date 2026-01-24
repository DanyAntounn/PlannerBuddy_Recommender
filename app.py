# app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from typing import Optional, Union
from chatbot import router as chatbot_router

import os
import requests

from recommender import (
    build_flutter_payload,
    enrich_collection_with_profiles_from_firestore,
    firestore_trip_generate,                # fallback (no query)
    firestore_trip_generate_from_query,     # query-driven
    rank_places_from_firestore,
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
