from collections import defaultdict

def compute_node_pip_totals(board):
    def number_to_pips(number):
        pip_map = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
        return pip_map.get(number, 0)
    node_to_tiles = defaultdict(list)
    for _, tile in board.map.land_tiles.items():
        for node_id in tile.nodes.values():
            node_to_tiles[node_id].append(tile)
    totals = {
        node_id: sum(
            number_to_pips(tile.number) if isinstance(tile.number, int) else 0
            for tile in tiles
        )
        for node_id, tiles in node_to_tiles.items()
    }
    sorted_totals = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    for node_id, total in sorted_totals:
        print(f"node id {node_id} has total adjacent number value {total}")
    return sorted_totals
