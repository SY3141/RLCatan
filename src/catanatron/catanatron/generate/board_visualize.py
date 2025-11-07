import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from catanatron.models.board import Board
from catanatron.models.map import TOURNAMENT_MAP
import random
import os


RESOURCE_COLORS = {
    "WOOD": "#2e8b57",  # green
    "BRICK": "#b22222",  # red
    "SHEEP": "#9acd32",  # light green
    "WHEAT": "#f0e68c",  # yellow
    "ORE": "#808080",  # gray
    None: "#d2b48c",  # desert
}


# === Geometry helpers ===
def hex_to_pixel(q, r, size=1):
    """
    Axial (q, r) → Cartesian (x, y) for pointy-topped hex grid.
    size = distance from center to vertex.
    This spacing guarantees edge sharing (no gaps).
    """
    x = size * math.sqrt(3) * (q + r / 2)
    y = size * 1.5 * r
    return x, y


def hex_corners(x, y, size):
    """Return vertices of a pointy-topped hex centered at (x, y)."""
    corners = []
    for i in range(6):
        angle_deg = 60 * i - 30  # pointy top
        angle_rad = math.radians(angle_deg)
        corners.append((x + size * math.cos(angle_rad), y + size * math.sin(angle_rad)))
    return corners


# === Draw board ===


def generate_board_image(board: Board):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")

    size = 1.0  # distance center→vertex
    for coords, tile in board.map.land_tiles.items():
        q, r, s = coords
        x, y = hex_to_pixel(q, r, size)

        resource = getattr(tile.resource, "name", tile.resource)
        color = RESOURCE_COLORS.get(resource, RESOURCE_COLORS[None])

        # Draw hex
        corners = hex_corners(x, y, size)
        hex_patch = mpatches.Polygon(
            corners, closed=True, edgecolor="black", facecolor=color, lw=1.8
        )
        ax.add_patch(hex_patch)

        # Draw number token
        number = tile.number or ""
        if number:
            circ = plt.Circle((x, y), 0.35, color="white", ec="black", lw=1.3, zorder=3)
            ax.add_patch(circ)
            ax.text(
                x,
                y,
                str(number),
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        # Resource label (optional)
        ax.text(
            x,
            y - 0.5,
            resource or "DESERT",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    # === Layout ===
    ax.axis("equal")
    ax.axis("off")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)

    output_dir = os.path.join(os.path.dirname(__file__), "boards")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"board{seed}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved board image to {output_path}")
    plt.close(fig)
    # plt.show()
