from pathlib import Path
from PIL import Image

TARGET_SIZE = (100, 100)
CAPTURED_DIR = Path(__file__).parent / "captured"
RESIZED_DIR = Path(__file__).parent / "resized"


def resize_images() -> None:
    if not CAPTURED_DIR.exists():
        raise FileNotFoundError(f"Captured directory not found: {CAPTURED_DIR}")

    RESIZED_DIR.mkdir(exist_ok=True)

    image_paths = sorted(CAPTURED_DIR.glob("*.png"))
    if not image_paths:
        print(f"No PNG images found in {CAPTURED_DIR}")
        return

    total = len(image_paths)
    processed = 0

    for image_path in image_paths:
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                resized = img.resize(TARGET_SIZE, resample=Image.Resampling.LANCZOS)

                output_path = RESIZED_DIR / image_path.name
                resized.save(output_path)
                processed += 1
                print(f"Saved {output_path.name}")
        except Exception as exc:
            print(f"Failed to process {image_path.name}: {exc}")


if __name__ == "__main__":
    resize_images()
