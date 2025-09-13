from fastapi import FastAPI, Body
from recommender import recommend_food_places, recommend_activity_places

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Recommender API is running"}

@app.post("/recommend/food")
def food_recommendation(user_profile: dict = Body(...), food_places: list = Body(...)):
    return recommend_food_places(user_profile, food_places)

@app.post("/recommend/activity")
def activity_recommendation(user_profile: dict = Body(...), activity_places: list = Body(...)):
    return recommend_activity_places(user_profile, activity_places)
