"""Download and verify Kaggle Cinema Tickets dataset."""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
import yaml

DATASET_SLUG = "arashnic/cinema-ticket"
RAW_DIR = Path("data/raw")
EXPECTED_CSV = RAW_DIR / "cinemaTicket_Ref.csv"
CONFIG_PATH = Path("config.yaml")


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file in streaming chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def update_config_sha256(computed_hash: str) -> None:
    """Store or verify SHA-256 hash in config.yaml."""
    if not CONFIG_PATH.exists():
        print(f"Warning: {CONFIG_PATH} not found; skipping hash storage.")
        return

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    data_cfg = config.get("data", {})
    existing_hash = data_cfg.get("sha256", "").strip()

    if not existing_hash:
        print(f"Storing newly computed SHA-256 in {CONFIG_PATH}: {computed_hash}")
        data_cfg["sha256"] = computed_hash
        config["data"] = data_cfg
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    else:
        if existing_hash == computed_hash:
            print(f"SHA-256 verification successful: {computed_hash}")
        else:
            print(f"WARNING: SHA-256 mismatch! Expected in config: {existing_hash}, but found: {computed_hash}")


def find_kaggle_cmd():
    """Find Kaggle executable in active venv, PATH, or python -m kaggle."""
    # Check venv Scripts
    venv_kaggle = Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle")
    if venv_kaggle.exists():
        return [str(venv_kaggle)]
    which_kaggle = shutil.which("kaggle")
    if which_kaggle:
        return [which_kaggle]
    # Fallback to python module
    return [sys.executable, "-m", "kaggle"]


def download():
    """Download and extract dataset via Kaggle CLI or verify existing."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if EXPECTED_CSV.exists():
        print(f"Dataset already exists at {EXPECTED_CSV}. Skipping download.")
    else:
        print(f"Downloading dataset '{DATASET_SLUG}' to {RAW_DIR}...")
        kaggle_cmd = find_kaggle_cmd()
        cmd = kaggle_cmd + ["datasets", "download", "-d", DATASET_SLUG, "-p", str(RAW_DIR), "--unzip"]
        print(f"Running command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
        except Exception as e:
            err_msg = ""
            if hasattr(e, "stderr") and e.stderr:
                err_msg = e.stderr
            elif hasattr(e, "stdout") and e.stdout:
                err_msg = e.stdout
            print(f"\n[ERROR] Kaggle CLI download failed: {e}\n{err_msg}")
            print("\n" + "=" * 70)
            print("MANUAL FALLBACK INSTRUCTIONS:")
            print("1. Download the dataset zip manually from:")
            print("   https://www.kaggle.com/datasets/arashnic/cinema-ticket")
            print("2. Extract 'cinemaTicket_Ref.csv' into 'data/raw/cinemaTicket_Ref.csv'")
            print("3. Re-run this script: python scripts/download_data.py")
            print("=" * 70 + "\n")
            sys.exit(1)

        # Check if CSV exists (case-insensitive search if needed)
        if not EXPECTED_CSV.exists():
            # Check if extracted with different casing
            candidates = list(RAW_DIR.glob("*.csv"))
            match = next((c for c in candidates if c.name.lower() == "cinematicket_ref.csv"), None)
            if match:
                print(f"Found file with name {match.name}, renaming to {EXPECTED_CSV.name}")
                match.rename(EXPECTED_CSV)
            else:
                print(f"[ERROR] Download finished but {EXPECTED_CSV} was not found among {candidates}.")
                print("Please check data/raw/ and verify the file name.")
                sys.exit(1)

    # Compute SHA-256
    file_hash = compute_sha256(EXPECTED_CSV)
    print(f"File: {EXPECTED_CSV}")
    print(f"Size: {EXPECTED_CSV.stat().st_size:,} bytes")
    print(f"SHA-256: {file_hash}")
    update_config_sha256(file_hash)
    print("Download and verification complete.")


if __name__ == "__main__":
    download()

