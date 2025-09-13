from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from recommender import recommend_food_places, recommend_activity_places

app = FastAPI()

class FoodRequest(BaseModel):
    user_profile: Dict
    food_places: List[Dict]

class ActivityRequest(BaseModel):
    user_profile: Dict
    activity_places: List[Dict]

@app.get("/")
def root():
    return {"message": "Recommender API is running"}

@app.post("/recommend/food")
def food_recommendation(request: FoodRequest):
    return recommend_food_places(request.user_profile, request.food_places)

@app.post("/recommend/activity")
def activity_recommendation(request: ActivityRequest):
    return recommend_activity_places(request.user_profile, request.activity_places)
