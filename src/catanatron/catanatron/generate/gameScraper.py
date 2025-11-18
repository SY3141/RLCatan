import requests
from catanatron.models.board import Board
from board_visualize import generate_board_image
from pandas import read_csv
from catanatron.models.map import CatanMap, BASE_MAP_TEMPLATE
from catanatron.models.map import WOOD, BRICK, SHEEP, WHEAT, ORE

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

tile_number_map = {
    0: 13,
    1: 12,
    2: 11,
    3: 10,
    4: 9,
    5: 8,
    6: 7,
    7: 18,
    8: 17,
    9: 16,
    10: 15,
    11: 14,
    12: 4,
    13: 3,
    14: 2,
    15: 1,
    16: 6,
    17: 5,
    18: 0,
}

def board_setup_to_catan_map(board_setup):
    """Convert a scraped `board_setup` dict into a CatanMap instance.

    `board_setup` is expected to map integer tile_id -> {"type": <str>, "number": <int|None>}.
    Tile ids correspond to the LandTile.id numbering used by the map generator.
    """
    
    # map the scraped type strings to the FastResource constants used by the
    # codebase. If we encounter an unknown string, leave it as None (desert/water).
    str_to_res = {0: None, 1: WOOD, 2: BRICK, 3: SHEEP, 4: WHEAT, 5: ORE}

    # Start from a template map to get the correct topology (nodes/edges objects).
    base_map = CatanMap.from_template(BASE_MAP_TEMPLATE)

    # Update land tiles by tile id using the scraped board_setup.
    # board_setup keys should align with LandTile.id values.
    for tile_id, tile in list(base_map.tiles_by_id.items()):
        entry = board_setup.get(tile_id)
        if entry is None:
            # no scraped info for this tile; leave as-is
            continue

        ttype = entry.get("type")
        number = entry.get("number")

        # normalize type string to lowercase if present
        if isinstance(ttype, str):
            ttype_norm = ttype.lower()
        else:
            ttype_norm = ttype

        resource = str_to_res.get(ttype_norm, None)

        # Assign resource and number onto tile object
        tile.resource = resource
        # If desert type, ensure number is None
        if resource is None:
            tile.number = None
        else:
            tile.number = number

    # Rebuild a CatanMap from the modified tiles dict so caches (node_production, etc.) are correct
    new_map = CatanMap.from_tiles(base_map.tiles)
    return new_map


if __name__ == "__main__":
    slugs = read_csv("replays.csv").squeeze().tolist()
    for slug in slugs:
        print("Processing slug:", slug)
        try:
            data = get_colonist_replay(slug)
            # Type is ['desert','wood', 'brick', 'sheep', 'wheat', 'ore'] = [0,1,2,3,4,5]
            board_setup_data = data["data"]["eventHistory"]["initialState"]["mapState"][
                "tileHexStates"
            ]
            board_setup = {}
            for tile_id, tile in board_setup_data.items():
                board_setup[tile_number_map[int(tile_id)]] = {
                    "type": tile.get("type"),
                    "number": tile.get("diceNumber"),
                }

            board_setup = dict(sorted(board_setup.items(), key=lambda item: item[0]))
            #for tile_id, tile in board_setup.items():
                #print(f"- {tile_id}, type: {tile['type']}, number: {tile['number']}")
            cmap = board_setup_to_catan_map(board_setup)
            for cid, tile in cmap.tiles_by_id.items():
                res = getattr(tile.resource, "name", tile.resource)
                #print(f"id={cid}, resource={res}, number={tile.number}")
            c_board = Board(catan_map=cmap)
            generate_board_image(c_board, slug)
        except Exception as e:
            print("Conversion to CatanMap failed:", e)
