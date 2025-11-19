import requests
from pandas import read_csv

def fetch_json(slug):
    url = f"https://colonist.io/api/replay/data-from-slug?replayUrlSlug={slug}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def print_top_level_keys(data):
    print("Top-level keys:")
    for key in data["data"]["eventHistory"]["events"][0]["stateChange"]["mapState"]["tileCornerStates"]:
        print(f"- {key}")


def main():
    #slug = ""
    #print_top_level_keys(data)
    slugs = read_csv("replays.csv").squeeze().tolist()
    for slug in slugs:
        print("Processing slug:", slug)
        data = fetch_json(slug)
        placement_data = []
        roads, buildings = 0, 0
        turn_index = 0
        while roads + buildings < 8:
            move = data["data"]["eventHistory"]["events"][turn_index]
            if "mapState" in move.get("stateChange", {}):
                if "tileCornerStates" in move["stateChange"]["mapState"]:
                    placement_data.append("building:" + ",".join(move['stateChange']['mapState']['tileCornerStates'].keys()))
                    buildings += 1
                elif "tileEdgeStates" in move["stateChange"]["mapState"]:
                    placement_data.append("road:" + ",".join(move['stateChange']['mapState']['tileEdgeStates'].keys()))
                    roads += 1
            turn_index += 1
        with open(f"placements_data/{slug}.txt", "w") as f:
            for entry in placement_data:
                f.write(entry + "\n")



if __name__ == "__main__":
    main()
