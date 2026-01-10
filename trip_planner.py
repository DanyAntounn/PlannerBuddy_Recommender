# planners/trip_planner.py
from nlp import extract_requirements_with_openai
from firestore import rank_places_from_firestore

def build_flutter_payload(plan_steps, public_base_url=""):
    out = []
    for step in plan_steps:
        places = []
        for rec in step["recommendations"]:
            profile = rec["profile"]
            entry = {
                "name": rec["name"],
                "score": round(rec["score"], 2),
                "rating": profile.get("rating", 0.0),
            }

            if profile.get("google_place_id") and public_base_url:
                entry["image_url"] = f"{public_base_url}/photo?place_id={profile['google_place_id']}"

            places.append(entry)

        out.append({"requirement": step["requirement"], "places": places})
    return out

def firestore_trip_generate_from_query(
    query,
    user_food,
    user_act,
    user_latitude=None,
    user_longitude=None,
):
    extraction = extract_requirements_with_openai(query)
    plan = []

    for req in extraction.get("requirements", []):
        typ = req.get("type")
        count = int(req.get("count", 1))
        kw = " ".join(req.get("keywords", []))

        if typ == "restaurant":
            recs = rank_places_from_firestore(
                "RestaurantsFinal",
                user_food,
                "food",
                count,
                kw,
                user_latitude,
                user_longitude
            )
        else:
            recs = rank_places_from_firestore(
                "ActivitiesFinal",
                user_act,
                "activity",
                count,
                kw,
                user_latitude,
                user_longitude
            )

        plan.append({
            "requirement": req,
            "recommendations": recs
        })

    return {"extracted": extraction, "plan": plan}
