"""Download the competition dataset using kagglehub with authentication handling."""
import os
import sys
import shutil
from pathlib import Path

def download():
    import kagglehub
    from kagglehub.config import get_kaggle_credentials, set_kaggle_credentials, set_kaggle_api_token

    competition_name = "iith-foundations-of-machine-learning-2026-hackathon"

    # Check if already authenticated
    creds = get_kaggle_credentials()
    if not creds:
        print("Kaggle credentials not detected.")
        print("You can get your Kaggle API token from: https://www.kaggle.com/settings/api")
        token_input = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
        if not token_input:
            username = input("Enter your Kaggle Username (or press Enter to enter API token directly): ").strip()
            if username:
                api_key = input("Enter your Kaggle API Key: ").strip()
                set_kaggle_credentials(username, api_key)
            else:
                api_token = input("Enter your Kaggle API token: ").strip()
                set_kaggle_api_token(api_token)

    print(f"\nDownloading dataset for '{competition_name}' via kagglehub...")
    try:
        path = kagglehub.competition_download(competition_name)
        print(f"Dataset successfully downloaded to: {path}")

        # Copy files to local data/ directory
        dest_dir = Path(__file__).resolve().parent / "data"
        dest_dir.mkdir(exist_ok=True)

        for item in os.listdir(path):
            src_path = os.path.join(path, item)
            dest_path = dest_dir / item
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dest_path)
                print(f"Copied {item} -> data/{item}")
            elif os.path.isdir(src_path):
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.copytree(src_path, dest_path)
                print(f"Copied directory {item} -> data/{item}")

        print("\nAll dataset files are ready in data/ folder!")
        return str(dest_dir)
    except Exception as e:
        print(f"\nError downloading data: {e}", file=sys.stderr)
        print("\nAlternative: You can manually download data from:")
        print(f"https://www.kaggle.com/competitions/{competition_name}/data")
        print("and place the extracted files directly into the data/ folder.")
        sys.exit(1)

if __name__ == "__main__":
    download()
