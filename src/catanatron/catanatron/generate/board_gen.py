from catanatron.models.board import Board
from catanatron.models.map import TOURNAMENT_MAP
import random


random.seed(19)
b_tour = Board()
# b_tour = Board(catan_map=TOURNAMENT_MAP)


# print("All tiles:", len(b_tour.map.tiles))
# print("Land tiles:", len(b_tour.map.land_tiles))


# for coords, tile in b_tour.map.land_tiles.items():
#     print(coords, "tile id", tile.id, "resource", getattr(tile.resource, "name", tile.resource), "number", tile.number)


for node_id, node in b_tour.map.nodes.items():
    # Each node knows what tiles it's next to
    adjacent_tiles = b_tour.map.adjacent_tiles[node_id]
    # Each node also knows what edges connect to it
    adjacent_edges = b_tour.map.adjacent_edges[node_id]

    print(f"Node {node_id}:")
    print(f"  Adjacent tiles: {adjacent_tiles}")
    print(f"  Adjacent edges: {adjacent_edges}")
    print()
