# app.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
from recommender import custom_trip_generate, auto_trip_generate, build_flutter_payload, enrich_collection_with_profiles_from_firestore

app = FastAPI(title="Trip Planner API")

class UserProfile(BaseModel):
    preferred_primary_features: list[str]
    ambiance: list[str]
    preferred_secondary_features: list[str]
    rating_threshold: float
    min_reviews_count: int

class CustomTripRequest(BaseModel):
    query: str
    user_food: UserProfile
    user_act: UserProfile
    location: str = "Beirut, Lebanon"

@app.post("/custom-trip")
def custom_trip(req: CustomTripRequest):
    res = custom_trip_generate(req.query, req.user_food.dict(), req.user_act.dict(), req.location)
    flutter_payload = build_flutter_payload(res["plan"])
    return {"extracted": res["extracted"], "plan": flutter_payload}

class AutoTripRequest(BaseModel):
    user_food: UserProfile
    user_act: UserProfile
    location: str = "Beirut, Lebanon"

@app.get("/enrich_firestore")
def enrich_firestore():
    enrich_collection_with_profiles_from_firestore("RestaurantsFinal", "food")
    enrich_collection_with_profiles_from_firestore("ActivitiesFinal", "activity")
    return {"status": "OK"}

@app.post("/auto-trip")
def auto_trip(req: AutoTripRequest):
    res = auto_trip_generate(req.user_food.dict(), req.user_act.dict(), req.location)
    flutter_payload = build_flutter_payload(res["plan"])
    return {"plan": flutter_payload}
