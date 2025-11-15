import requests

def fetch_json(slug):
    url = f"https://colonist.io/api/replay/data-from-slug?replayUrlSlug={slug}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def print_top_level_keys(data):
    print("Top-level keys:")
    for key in data["data"]["eventHistory"]["initialState"]["mapState"]["tileHexStates"].keys():
        print(f"- {key}")

def main():
    slug = "BPPRTG5zMVmn9gMX"
    data = fetch_json(slug)
    print_top_level_keys(data)

if __name__ == "__main__":
    main()
