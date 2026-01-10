# api.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from trip_planner import firestore_trip_generate_from_query, build_flutter_payload
from pydantic import BaseModel
from typing import Optional, Union

app = FastAPI()

class UserProfile(BaseModel):
    preferred_primary_features: list[str]
    ambiance: list[str]
    preferred_secondary_features: list[str]
    rating_threshold: float
    min_reviews_count: int

class TripRequest(BaseModel):
    query: Optional[str] = None
    user_food: UserProfile
    user_act: UserProfile
    num_restaurants: int = 1
    num_activities: int = 2
    location: Optional[Union[str, LocationPayload]] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

@app.post("/firestore-trip")
def firestore_trip(req: TripRequest):
    res = firestore_trip_generate_from_query(
        req.query,
        req.user_food.dict(),
        req.user_act.dict(),
        req.latitude,
        req.longitude
    )
    return {
        "extracted": res["extracted"],
        "plan": build_flutter_payload(res["plan"])
    }
