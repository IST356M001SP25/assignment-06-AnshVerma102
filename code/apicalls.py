import requests
from typing import Any, Dict

# Put your CENT Ischool IoT Portal API KEY here.
APIKEY = "bb522f26251fc11bc55e8944"
BASE_URL = "https://cent.ischool-iot.net/api"

# One session for all calls (connection pooling + shared headers)
_session = requests.Session()
_session.headers.update({"X-API-KEY": APIKEY})

def _call_api(
    method: str,
    path: str,
    params: Dict[str, Any] = None,
    data:  Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Internal helper for GET/POST to our IoT portal.
    """
    url = f"{BASE_URL}/{path}"
    response = _session.request(method, url, params=params, data=data)
    response.raise_for_status()
    return response.json()

def get_google_place_details(google_place_id: str) -> dict:
    return _call_api(
        "GET",
        "google/places/details",
        params={"place_id": google_place_id}
    )

def geocode(place: str) -> dict:
    return _call_api(
        "GET",
        "google/geocode",
        params={"location": place}
    )

def get_weather(lat: float, lon: float) -> dict:
    return _call_api(
        "GET",
        "weather/current",
        params={"lat": lat, "lon": lon, "units": "imperial"}
    )

def get_azure_sentiment(text: str) -> dict:
    return _call_api(
        "POST",
        "azure/sentiment",
        data={"text": text}
    )

def get_azure_key_phrase_extraction(text: str) -> dict:
    return _call_api(
        "POST",
        "azure/keyphrasextraction",
        data={"text": text}
    )

def get_azure_named_entity_recognition(text: str) -> dict:
    return _call_api(
        "POST",
        "azure/entityrecognition",
        data={"text": text}
    )
