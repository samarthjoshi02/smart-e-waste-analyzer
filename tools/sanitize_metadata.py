from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1] / "dataset" / "images"


def sanitize(path: Path) -> None:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            image.save(path, format="JPEG", quality=95, optimize=True)
        elif path.suffix.lower() == ".png":
            image.save(path, format="PNG", optimize=True)


def main() -> None:
    files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    for path in files:
        sanitize(path)
    print(f"Sanitized metadata from {len(files)} images")


if __name__ == "__main__":
    main()
