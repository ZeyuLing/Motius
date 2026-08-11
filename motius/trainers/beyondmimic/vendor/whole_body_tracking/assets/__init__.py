import os

# Conveniences to other module directories via relative paths
ASSET_DIR = os.path.abspath(
    os.environ.get("MOTIUS_BEYONDMIMIC_ASSET_ROOT", os.path.dirname(__file__))
)
