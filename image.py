from pathlib import Path

from PIL import Image


def load(path: Path) -> Image.Image:
    # Load image
    image = Image.open(path).convert("RGBA")

    # Create white background
    background = Image.new("RGBA", image.size, (255, 255, 255))
    combined = Image.alpha_composite(background, image).convert("RGB")

    return combined
