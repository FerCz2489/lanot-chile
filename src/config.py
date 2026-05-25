from pathlib import Path
import os

# ==========================================================
# DIRECTORIO BASE DEL PROYECTO
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# ==========================================================
# DATA ROOT
# ==========================================================
# Puede venir desde variable de entorno.
# Si no existe, usa ./data
# ==========================================================

LANOT_DATA = Path(
    os.getenv(
        "LANOT_DATA",
        BASE_DIR / "data"
    )
)

# ==========================================================
# INPUTS
# ==========================================================

RAW_GOES = LANOT_DATA / "raw" / "GOES"
RAW_S5P = LANOT_DATA / "raw" / "S5P"

# ==========================================================
# OUTPUTS
# ==========================================================

OUT_RGB_ASH = LANOT_DATA / "outputs" / "rgb_ash"
OUT_SO2 = LANOT_DATA / "outputs" / "so2"

# ==========================================================
# CREAR DIRECTORIOS SI NO EXISTEN
# ==========================================================

for p in [
    RAW_GOES,
    RAW_S5P,
    OUT_RGB_ASH,
    OUT_SO2,
]:
    p.mkdir(parents=True, exist_ok=True)
