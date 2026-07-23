import os
import urllib.request
from pathlib import Path
from zipfile import ZipFile

MODELS = {
    "craft_mlt_25k.zip": {
        "url": "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip",
        "file": "craft_mlt_25k.pth",
    },
    "english_g2.zip": {
        "url": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip",
        "file": "english_g2.pth",
    },
}


def main():
    # EasyOCR looks for models in ~/.EasyOCR/model
    dest_dir = Path.home() / ".EasyOCR" / "model"
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target directory for models: {dest_dir}")

    for zip_name, info in MODELS.items():
        pth_path = dest_dir / info["file"]
        if pth_path.exists():
            print(f"{info['file']} already exists. Skipping download.")
            continue

        zip_path = dest_dir / zip_name
        url = info["url"]

        print(f"Downloading {zip_name} from {url}...")
        try:
            # Configure custom headers to avoid any bot blockages
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
                # Copy with progress
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                block_size = 8192
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    if total_size:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end="")
                print()

            print(f"Extracting {zip_name}...")
            with ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extract(info["file"], dest_dir)

            print(f"Finished setting up {info['file']}.")
        except Exception as e:
            print(f"Error handling {zip_name}: {e}")
        finally:
            if zip_path.exists():
                os.remove(zip_path)

    print("EasyOCR models download & extraction process complete.")


if __name__ == "__main__":
    main()
