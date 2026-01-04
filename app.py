from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Union

from recommender import (
    build_flutter_payload,
    enrich_collection_with_profiles_from_firestore,
    firestore_trip_generate,                # fallback (no query)
    firestore_trip_generate_from_query,     # query-driven
)

app = FastAPI(title="Trip Planner API")


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
    # Flutter may send query
    query: Optional[str] = None

    user_food: UserProfile
    user_act: UserProfile

    # Defaults (used only if query is empty)
    num_restaurants: int = 1
    num_activities: int = 2

    # Flutter sometimes sends location as string OR object; we accept both and read lat/lon from it if present.
    location: Optional[Union[str, LocationPayload]] = None

    # Also allow top-level coords (compat)
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@app.get("/enrich_firestore")
def enrich_firestore():
    enrich_collection_with_profiles_from_firestore("RestaurantsFinal", "food")
    enrich_collection_with_profiles_from_firestore("ActivitiesFinal", "activity")
    return {"status": "OK"}


@app.post("/firestore-trip")
def firestore_trip(req: FirestoreTripRequest):
    # Resolve user lat/lon from either:
    # 1) top-level latitude/longitude
    # 2) location.lat/location.lon (Flutter payload you showed)
    user_lat = req.latitude
    user_lon = req.longitude

    if (user_lat is None or user_lon is None) and isinstance(req.location, LocationPayload):
        user_lat = req.location.lat
        user_lon = req.location.lon

    # Query-driven
    if req.query and req.query.strip():
        res = firestore_trip_generate_from_query(
            query=req.query,
            user_food=req.user_food.dict(),
            user_act=req.user_act.dict(),
            user_latitude=user_lat,
            user_longitude=user_lon,
        )
        flutter_payload = build_flutter_payload(res["plan"])
        return {"extracted": res["extracted"], "plan": flutter_payload}

    # Fallback (no query)
    num_restaurants = req.num_restaurants if req.num_restaurants > 0 else 1
    num_activities = req.num_activities if req.num_activities > 0 else 2

    res = firestore_trip_generate(
        req.user_food.dict(),
        req.user_act.dict(),
        num_restaurants=num_restaurants,
        num_activities=num_activities,
        location="Lebanon",  # metadata only
        user_latitude=user_lat,
        user_longitude=user_lon,
    )
    flutter_payload = build_flutter_payload(res["plan"])
    return {"plan": flutter_payload}
