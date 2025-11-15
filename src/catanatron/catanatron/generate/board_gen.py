from catanatron.models.board import Board
from catanatron.models.map import TOURNAMENT_MAP
import random
from collections import defaultdict

from game_gen import compute_node_pip_totals

random.seed(19)
b_tour = Board()

def number_to_pips(number):
    if number in [6, 8]:
        return 5
    elif number in [5, 9]:
        return 4
    elif number in [4, 10]:
        return 3
    elif number in [3, 11]:
        return 2
    elif number in [2, 12]:
        return 1
    else:
        return 0


node_to_tiles = defaultdict(list)
for coords, tile in b_tour.map.land_tiles.items():
    for node_id in tile.nodes.values():
        node_to_tiles[node_id].append(tile)

totals = defaultdict(int)
wheats = defaultdict(int)
ores = defaultdict(int)
for node_id, tiles in node_to_tiles.items():
    print(f"node id {node_id} adjacent tiles:")
    total = 0
    for tile in tiles:
        resource_name = getattr(tile.resource, "name", tile.resource)
        print(f"  tile id {tile.id} | resource {resource_name} | number {tile.number}")
        total += number_to_pips(tile.number) if isinstance(tile.number, int) else 0
    totals[node_id] = total

sorted_totals = sorted(totals.items(), key=lambda x: x[1], reverse=True)
# for node_id, total in sorted_totals:
#     print(f"node id {node_id} has total adjacent number value {total}")
