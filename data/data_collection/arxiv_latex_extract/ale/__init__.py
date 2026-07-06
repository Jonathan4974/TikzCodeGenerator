from os import makedirs
from os.path import join, dirname, abspath
import yaml

SCRIPT_DIR = dirname(abspath(__file__))
CFG_PATH = join(SCRIPT_DIR, "..", "..", "collection_config.yaml")
_global_config = yaml.safe_load(open(CFG_PATH, "r"))

BASE_DIR = _global_config.get("base_dir", "/usr/prakt/s0042/projects/data/arxiv_TikZ_dataset/2510_2603")
ARCHIVE_DIR = join(BASE_DIR, "archives")
LATEX_DIR = join(BASE_DIR, "extracted")

# ARCHIVE_DIR = "archives"
# LATEX_DIR = "extracted"
ARXIV_URL = "https://arxiv.org/abs/"

for path in [ARCHIVE_DIR, LATEX_DIR]:
    makedirs(path, exist_ok=True)
