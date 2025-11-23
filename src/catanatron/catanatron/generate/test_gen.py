# add temporarily in test or local run
import random
from catanatron.models.map import CatanMap, BASE_MAP_TEMPLATE, number_probability

random.seed(123)
cmap = CatanMap.from_template(BASE_MAP_TEMPLATE)

# print tile coordinates -> (resource, number)
for coord, tile in sorted(cmap.land_tiles.items(), key=lambda x: (x[0])):
    print(
        coord,
        tile.id,
        tile.resource,
        tile.number,
        number_probability(tile.number) if tile.number else None,
    )
