# LANOT - Detección de SO2 y ASH

Proyecto para análisis de datos satelitales (Sentinel-5P) enfocado en detección de SO₂.

## Estructura

- src/: scripts principales
- data/: datos (raw, processed, outputs)
- notebooks/: pruebas
- venv/: entorno virtual

## Uso

Activar entorno:

source venv/bin/activate

# Construir contenedor
sudo apptainer build lanot-chile.sif lanot-chile.def

# SO2
export COPERNICUS_USER="..."
export COPERNICUS_PASSWORD="..."
apptainer exec lanot-chile.sif python src/sentinel5p.py

# Ceniza GOES
apptainer exec lanot-chile.sif python src/ash.py
