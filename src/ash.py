import os
import re
import glob
from datetime import datetime, timedelta

import boto3
from botocore import UNSIGNED
from botocore.config import Config

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer


# ==========================================================
# CONFIGURACIÓN GENERAL
# ==========================================================

INPUT_BASE = "/home/fercz/lanot/data/raw/GOES"
OUTPUT_BASE = "/home/fercz/lanot/data/outputs/rgb_ash"

# Para eventos históricos de Chile:
BUCKET_GOES = "noaa-goes16"

# Producto ABI Full Disk, canal individual por archivo
PRODUCTO_GOES = "ABI-L2-CMIPF"

# Incluye C07 para el RGB microfísica/Pavolonis
BANDAS_REQUERIDAS = ["C07", "C11", "C13", "C14", "C15"]

EVENTOS = [
    "2019-03-08",
    "2020-01-30",
    "2020-04-08",
    "2022-12-10",
    "2023-01-27",
]

VOLCANES = {
    "Chillan": {
        "lat": -36.868,
        "lon": -71.378,
        "bbox": [-73.5, -38.2, -69.8, -35.2],
    },
    "Lascar": {
        "lat": -23.370,
        "lon": -67.730,
        "bbox": [-70.0, -25.0, -65.5, -21.5],
    },
    "Villarrica": {
        "lat": -39.420,
        "lon": -71.930,
        "bbox": [-74.0, -41.0, -70.0, -38.0],
    },
}


# ==========================================================
# AWS S3 DESCARGA AUTOMÁTICA
# ==========================================================

def fecha_a_juliano(fecha):
    dt = datetime.strptime(fecha, "%Y-%m-%d")
    return dt.year, dt.timetuple().tm_yday


def descargar_goes_eventos(horas_utc=range(0, 24)):
    s3 = boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED)
    )

    for fecha in EVENTOS:
        year, jday = fecha_a_juliano(fecha)

        for hour in horas_utc:
            prefix = f"{PRODUCTO_GOES}/{year}/{jday:03d}/{hour:02d}/"

            print(f"\nBuscando:")
            print(f"s3://{BUCKET_GOES}/{prefix}")

            try:
                resp = s3.list_objects_v2(
                    Bucket=BUCKET_GOES,
                    Prefix=prefix
                )
            except Exception as e:
                print(f"Error AWS: {e}")
                continue

            if "Contents" not in resp:
                print("No hay archivos.")
                continue

            for obj in resp["Contents"]:
                key = obj["Key"]
                nombre = os.path.basename(key)
                banda = extraer_banda(nombre)

                if banda not in BANDAS_REQUERIDAS:
                    continue

                out_dir = os.path.join(
                    INPUT_BASE,
                    fecha,
                    f"{hour:02d}",
                    banda
                )

                os.makedirs(out_dir, exist_ok=True)

                out_path = os.path.join(out_dir, nombre)

                if os.path.exists(out_path):
                    print(f"Ya existe: {nombre}")
                    continue

                print(f"Descargando: {nombre}")

                try:
                    s3.download_file(
                        BUCKET_GOES,
                        key,
                        out_path
                    )
                except Exception as e:
                    print(f"Error descarga: {e}")


# ==========================================================
# UTILIDADES
# ==========================================================

def normalizar(data, vmin=None, vmax=None):
    if vmin is None:
        vmin = np.nanmin(data)

    if vmax is None:
        vmax = np.nanmax(data)

    if vmax == vmin:
        return np.zeros_like(data)

    data = (data - vmin) / (vmax - vmin)
    data = np.clip(data, 0, 1)
    data = np.nan_to_num(data, nan=0.0)

    return data


def extraer_banda(path):
    nombre = os.path.basename(path)
    m = re.search(r"C(0[1-9]|1[0-6])", nombre)

    if m:
        return "C" + m.group(1)

    return None


def extraer_timestamp(path):
    nombre = os.path.basename(path)

    m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})", nombre)

    if m:
        year = int(m.group(1))
        jday = int(m.group(2))
        hour = int(m.group(3))
        minute = int(m.group(4))

        dt = datetime(year, 1, 1) + timedelta(days=jday - 1)
        dt = dt.replace(hour=hour, minute=minute, second=0)

        return dt.strftime("%Y-%m-%d_%H%M")

    return "timestamp_desconocido"


def buscar_y_agrupar_archivos(input_base):
    archivos = glob.glob(
        os.path.join(input_base, "**", "*.nc"),
        recursive=True
    )

    print("\nArchivos .nc encontrados:", len(archivos))

    grupos = {}

    for archivo in archivos:
        banda = extraer_banda(archivo)

        if banda not in BANDAS_REQUERIDAS:
            continue

        timestamp = extraer_timestamp(archivo)

        if timestamp not in grupos:
            grupos[timestamp] = {}

        grupos[timestamp][banda] = archivo

    grupos_completos = {
        t: bandas for t, bandas in grupos.items()
        if all(b in bandas for b in BANDAS_REQUERIDAS)
    }

    return grupos_completos


def recortar_goes_da(da, ds, bbox):
    lon_min, lat_min, lon_max, lat_max = bbox

    proj_attrs = ds["goes_imager_projection"].attrs
    h = proj_attrs["perspective_point_height"]

    crs_goes = CRS.from_cf(proj_attrs)
    crs_geo = CRS.from_epsg(4326)

    transformer = Transformer.from_crs(
        crs_geo,
        crs_goes,
        always_xy=True
    )

    x1, y1 = transformer.transform(lon_min, lat_min)
    x2, y2 = transformer.transform(lon_max, lat_max)

    x_min = min(x1, x2) / h
    x_max = max(x1, x2) / h
    y_min = min(y1, y2) / h
    y_max = max(y1, y2) / h

    if da.y[0] > da.y[-1]:
        da_crop = da.sel(
            x=slice(x_min, x_max),
            y=slice(y_max, y_min)
        )
    else:
        da_crop = da.sel(
            x=slice(x_min, x_max),
            y=slice(y_min, y_max)
        )

    return da_crop


def leer_banda_recortada(archivo, bbox):
    ds = xr.open_dataset(archivo)

    if "CMI" not in ds:
        raise ValueError(f"No hay variable CMI en: {archivo}")

    da = ds["CMI"].astype(float)
    da_crop = recortar_goes_da(da, ds, bbox)

    return da_crop.compute().values


# ==========================================================
# RGB
# ==========================================================

def crear_rgbs(c07, c11, c13, c14, c15):
    rNOAA = c15 - c13
    gNOAA = c14 - c11
    bNOAA = c13

    rgbNOAA = np.dstack([
        normalizar(rNOAA, -6.7, 2.6),
        normalizar(gNOAA, -6.0, 6.3),
        normalizar(bNOAA, 243.6, 302.4),
    ])

    rHOTVOLC = c13 - c15
    gHOTVOLC = c13 - c11
    bHOTVOLC = c13

    rgbHOTVOLC = np.dstack([
        normalizar(rHOTVOLC),
        normalizar(gHOTVOLC),
        normalizar(bHOTVOLC),
    ])

    rCNN = c15 - c13
    gCNN = c13 - c11
    bCNN = c13

    rgbCNN = np.dstack([
        normalizar(rCNN, -4, 2),
        normalizar(gCNN, -4, 5),
        normalizar(bCNN, 243, 303),
    ])

    rMICRO = c15 - c13
    gMICRO = c13 - c07
    bMICRO = c13

    rgbMICRO = np.dstack([
        normalizar(rMICRO),
        normalizar(gMICRO),
        normalizar(bMICRO),
    ])

    return {
        "NOAA_NASA": rgbNOAA,
        "HOTVOLC": rgbHOTVOLC,
        "CNN": rgbCNN,
        "MICROPHYSICS_PAVOLONIS": rgbMICRO,
    }


# ==========================================================
# PLOT
# ==========================================================

def guardar_rgb(
    rgb,
    out_path,
    titulo,
    bbox,
    lon_volcan,
    lat_volcan,
    nombre_volcan
):
    lon_min, lat_min, lon_max, lat_max = bbox

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(
        rgb,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin="upper"
    )

    ax.scatter(
        lon_volcan,
        lat_volcan,
        s=45,
        marker="^",
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        zorder=5
    )

    ax.text(
        lon_volcan + 0.08,
        lat_volcan + 0.08,
        nombre_volcan,
        fontsize=8,
        color="white",
        bbox=dict(
            facecolor="black",
            alpha=0.45,
            edgecolor="none"
        ),
        zorder=6
    )

    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ==========================================================
# PROCESAMIENTO PRINCIPAL
# ==========================================================

def main():
    descargar_goes_eventos(
        horas_utc=range(0, 24)
    )

    grupos = buscar_y_agrupar_archivos(INPUT_BASE)

    print(f"\nEscenas completas encontradas: {len(grupos)}")

    for timestamp, archivos_bandas in sorted(grupos.items()):

        if "-" in timestamp:
            fecha_timestamp = timestamp.split("_")[0]

            if fecha_timestamp not in EVENTOS:
                continue

        for nombre_volcan, info in VOLCANES.items():
            print(f"\nProcesando {nombre_volcan} | {timestamp}")

            bbox = info["bbox"]

            c07 = leer_banda_recortada(
                archivos_bandas["C07"],
                bbox
            )

            c11 = leer_banda_recortada(
                archivos_bandas["C11"],
                bbox
            )

            c13 = leer_banda_recortada(
                archivos_bandas["C13"],
                bbox
            )

            c14 = leer_banda_recortada(
                archivos_bandas["C14"],
                bbox
            )

            c15 = leer_banda_recortada(
                archivos_bandas["C15"],
                bbox
            )

            rgbs = crear_rgbs(
                c07,
                c11,
                c13,
                c14,
                c15
            )

            out_dir = os.path.join(
                OUTPUT_BASE,
                nombre_volcan,
                timestamp.split("_")[0],
                timestamp
            )

            os.makedirs(out_dir, exist_ok=True)

            for nombre_rgb, rgb in rgbs.items():
                titulo = (
                    f"{nombre_volcan} | "
                    f"{timestamp} UTC | "
                    f"{nombre_rgb}"
                )

                out_path = os.path.join(
                    out_dir,
                    f"{nombre_rgb}.png"
                )

                guardar_rgb(
                    rgb=rgb,
                    out_path=out_path,
                    titulo=titulo,
                    bbox=bbox,
                    lon_volcan=info["lon"],
                    lat_volcan=info["lat"],
                    nombre_volcan=nombre_volcan
                )

    print("\nProceso terminado.")
    print("Outputs en:", OUTPUT_BASE)


if __name__ == "__main__":
    main()