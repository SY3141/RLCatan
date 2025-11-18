import json
from pandas import read_csv
from gameScraper import get_colonist_replay
import os 
import requests

def get_colonist_replay(slug: str):
    url = f"https://colonist.io/api/replay/data-from-slug?replayUrlSlug={slug}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # raises error for non-200 responses
        data = response.json()
        return data

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e} (status code {response.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
    except ValueError:
        print("Failed to parse JSON — response was not JSON")


slugs = read_csv("replays.csv").squeeze().tolist()
print(slugs)
for slug in slugs:
    url = f"https://colonist.io/api/replay/data-from-slug?replayUrlSlug={slug}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()  # raises error for non-200 responses
    data = response.json()
    file_path = os.path.join("replays", f"{slug}.json")

    # Write JSON to file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)