import os
import sys
import warnings
import subprocess
import importlib

from google.colab import drive

from src.config import (
    PROJECT_ROOT,
    DATA_ZIP_PATH,
    EXTRACT_DIR,
    DATASET_DIR,
    OUTPUT_DIR,
    MODEL_DIR,
    TABLE_DIR,
    FIG_DIR,
)

warnings.filterwarnings("ignore")


def mount_drive():
    drive.mount("/content/drive")


def add_project_to_syspath():
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)


def install_if_missing(package_name, import_name=None):
    module_name = import_name if import_name else package_name
    try:
        importlib.import_module(module_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


def install_required_packages():
    install_if_missing("wfdb")
    install_if_missing("neurokit2")
    install_if_missing("scikit-learn", "sklearn")
    install_if_missing("tensorflow")
    install_if_missing("scipy")
    install_if_missing("pandas")
    install_if_missing("matplotlib")


def ensure_directories():
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def unzip_dataset_if_needed():
    if not os.path.exists(DATASET_DIR):
        subprocess.run(["unzip", "-q", DATA_ZIP_PATH, "-d", EXTRACT_DIR], check=True)


def prepare_environment():
    add_project_to_syspath()
    install_required_packages()
    ensure_directories()
    unzip_dataset_if_needed()

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATASET_DIR :", DATASET_DIR)
    print("OUTPUT_DIR  :", OUTPUT_DIR)
