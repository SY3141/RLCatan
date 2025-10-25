from catanatron.models.board import Board
from catanatron.models.map import TOURNAMENT_MAP
import random


random.seed(42) 
b_tour = Board(catan_map=TOURNAMENT_MAP)

for coords, tile in b_tour.map.land_tiles.items():
    print(coords, "tile id", tile.id, "resource", getattr(tile.resource, "name", tile.resource), "number", tile.number)
    