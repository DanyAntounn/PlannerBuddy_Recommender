# api.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from trip_planner import firestore_trip_generate_from_query, build_flutter_payload
from pydantic import BaseModel
from typing import Optional, Union
import os
import requests

app = FastAPI()

class UserProfile(BaseModel):
    preferred_primary_features: list[str]
    ambiance: list[str]
    preferred_secondary_features: list[str]
    rating_threshold: float
    min_reviews_count: int

class LocationPayload(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None

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
