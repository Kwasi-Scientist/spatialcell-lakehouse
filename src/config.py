from pathlib import Path
import yaml
from typing import Any


with open("configs/dataset_config.yaml", "r") as file:
    config = yaml.safe_load(file)



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


PROJECT_ROOT = Path(config["project_root"]).resolve()


CURRENT_FILE = Path(config["C:\Users\Kwaz9\Documents\coding\airflow\ELT\spatialcell-lakehouse\src\config.py"]).resolve()

SRC_DIR = Path(config["C:\Users\Kwaz9\Documents\coding\airflow\ELT\spatialcell-lakehouse\src"]).resolve()

REPO_ROOT = Path(config["C:\Users\Kwaz9\Documents\coding\airflow\ELT\spatialcell-lakehouse"]).resolve()

DEFAULT_CONFIG_PATH = REPO_ROOT / config["configs"]["dataset_config.yaml"]


def load_config(config_path: Path | str) -> dict:
    """
    Docstring for load_config
    
    :param config_path: Loads a YAML configuration file and returns its contents
    as a Python dictionary
    :type config_path: Path | str
    :return: Python dictionary of YAML contents
    :rtype: dict
    """
    #if no path use default config path
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    #verify that the file exists


    #read the yaml safely
    with open(config_path, "r") as file:
        yaml_config = yaml.safe_load(file)

    if yaml_config is null:
        raise ValueError("The configuration file is empty")
    
    if type(yaml_config) != dict:
        raise ValueError("The top-lvevl YAML structure must be a mapping")
    
    return yaml_config


def _validate_required_sections(yaml_config_dict):
    