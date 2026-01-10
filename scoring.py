# services/scoring.py
import math

def match_user_to_place(profile, place_profile, typ):
    score = 0

    if any(f in place_profile["primary_features"] for f in profile.get("preferred_primary_features", [])):
        score += 5

    score += len(set(profile.get("ambiance", [])) & set(place_profile["ambiance"])) * 3
    score += len(set(profile.get("preferred_secondary_features", [])) & set(place_profile["secondary_features"])) * 2

    if place_profile["avg_sentiment"] > 0.3:
        score += 3

    if place_profile["rating"] >= profile.get("rating_threshold", 0.0):
        score += 4

    if place_profile["num_reviews"] >= profile.get("min_reviews_count", 0):
        score += 2

    return score

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def distance_bias_bonus(user_lat, user_lon, place_lat, place_lon):
    if None in (user_lat, user_lon, place_lat, place_lon):
        return 0.0
    d = haversine_km(user_lat, user_lon, place_lat, place_lon)
    if d <= 2: return 1.5
    if d <= 5: return 1.0
    if d <= 10: return 0.6
    if d <= 20: return 0.3
    return 0.0

def diversify_by_distance(candidates, k, min_spread_km=2.0):
    chosen = []
    ordered = sorted(candidates, key=lambda x: x["score"], reverse=True)

    for c in ordered:
        if len(chosen) >= k:
            break

        clat = c["profile"].get("latitude")
        clon = c["profile"].get("longitude")

        ok = True
        for p in chosen:
            plat = p["profile"].get("latitude")
            plon = p["profile"].get("longitude")
            if None not in (clat, clon, plat, plon):
                if haversine_km(clat, clon, plat, plon) < min_spread_km:
                    ok = False
                    break

        if ok:
            chosen.append(c)

    if len(chosen) < k:
        for c in ordered:
            if c not in chosen:
                chosen.append(c)
                if len(chosen) >= k:
                    break

    return chosen
