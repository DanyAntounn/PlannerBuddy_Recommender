# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from recommender import (
    auto_trip_generate,
    build_flutter_payload,
    enrich_collection_with_profiles_from_firestore,
    firestore_trip_generate,                # fallback (no query)
    firestore_trip_generate_from_query,     # NEW (query-driven)
)

app = FastAPI(title="Trip Planner API")


class UserProfile(BaseModel):
    preferred_primary_features: list[str]
    ambiance: list[str]
    preferred_secondary_features: list[str]
    rating_threshold: float
    min_reviews_count: int


class AutoTripRequest(BaseModel):
    user_food: UserProfile
    user_act: UserProfile
    location: str = "Beirut, Lebanon"


@app.post("/auto-trip")
def auto_trip(req: AutoTripRequest):
    res = auto_trip_generate(
        req.user_food.dict(),
        req.user_act.dict(),
        req.location,
    )
    flutter_payload = build_flutter_payload(res["plan"])
    return {"plan": flutter_payload}


@app.get("/enrich_firestore")
def enrich_firestore():
    enrich_collection_with_profiles_from_firestore("RestaurantsFinal", "food")
    enrich_collection_with_profiles_from_firestore("ActivitiesFinal", "activity")
    return {"status": "OK"}


# ===== Firestore-based trip endpoint (Flutter should call this) =====

class FirestoreTripRequest(BaseModel):
    # NEW: Flutter sends query, so we accept it
    query: str | None = None

    user_food: UserProfile
    user_act: UserProfile

    # Keep these for backwards compatibility / fallback
    num_restaurants: int = 1
    num_activities: int = 2

    # Deprecated: keep accepting so Flutter doesn't break, but we ignore it
    location: str | None = None

    # User location (bias only)
    latitude: float | None = None
    longitude: float | None = None


@app.post("/firestore-trip")
def firestore_trip(req: FirestoreTripRequest):
    # Query-driven mode (your Flutter uses this)
    if req.query and req.query.strip():
        res = firestore_trip_generate_from_query(
            query=req.query,
            user_food=req.user_food.dict(),
            user_act=req.user_act.dict(),
            user_latitude=req.latitude,
            user_longitude=req.longitude,
        )
        flutter_payload = build_flutter_payload(res["plan"])
        return {"extracted": res["extracted"], "plan": flutter_payload}

    # Fallback mode if query is empty
    num_restaurants = req.num_restaurants if req.num_restaurants > 0 else 1
    num_activities = req.num_activities if req.num_activities > 0 else 2

    res = firestore_trip_generate(
        req.user_food.dict(),
        req.user_act.dict(),
        num_restaurants=num_restaurants,
        num_activities=num_activities,
        location="Beirut, Lebanon",  # ignored as a driver; only for metadata
        user_latitude=req.latitude,
        user_longitude=req.longitude,
    )
    flutter_payload = build_flutter_payload(res["plan"])
    return {"plan": flutter_payload}
