from pathlib import Path
import yaml


with open("configs/dataset_config.yaml", "r") as file:
    config = yaml.safe_load(file)

PROJECT_ROOT = Path(config["project_root"]).resolve()

BRONZE_DIR = PROJECT_ROOT / config["paths"]["bronze"]
SILVER_DIR = PROJECT_ROOT / config["paths"]["silver"]
GOLD_DIR = PROJECT_ROOT / config["paths"]["gold"]
REPORTS_DIR = PROJECT_ROOT / config["paths"]["reports"]

for directory in [
    BRONZE_DIR,
    SILVER_DIR,
    GOLD_DIR,
    REPORTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)