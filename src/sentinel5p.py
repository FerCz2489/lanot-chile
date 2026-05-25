from pathlib import Path
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
import os

import requests
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from config import RAW_S5P, OUT_SO2


# ============================================================
# CONFIGURACION GENERAL
# ============================================================

COPERNICUS_USER = os.getenv("COPERNICUS_USER")
COPERNICUS_PASSWORD = os.getenv("COPERNICUS_PASSWORD")

CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)

COLLECTION = "SENTINEL-5P"

# Sentinel-5P SO2 NRT suele venir en nombres tipo:
# S5P_NRTI_L2__SO2____YYYYMMDDT...
NAME_CONTAINS = "S5P_NRTI_L2__SO2"

# Buscar en los últimos N días
SEARCH_DAYS_BACK = 7

# Política de limpieza local
KEEP_DAYS = 7

# QA recomendado
QA_MIN = 0.5

# Visualización
USAR_ESCALA_FIJA = True
VMIN_FIJO = 0.0
VMAX_FIJO = 1.0e-2
PERCENTIL_VMAX = 99.0


# ============================================================
# VOLCANES
# ============================================================

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


# ============================================================
# COPERNICUS
# ============================================================

def obtener_token():
    if not COPERNICUS_USER or not COPERNICUS_PASSWORD:
        raise RuntimeError(
            "Faltan credenciales. Define COPERNICUS_USER y COPERNICUS_PASSWORD."
        )

    data = {
        "client_id": "cdse-public",
        "username": COPERNICUS_USER,
        "password": COPERNICUS_PASSWORD,
        "grant_type": "password",
    }

    r = requests.post(TOKEN_URL, data=data, timeout=60)
    r.raise_for_status()

    return r.json()["access_token"]


def buscar_producto_so2_mas_reciente():
    ahora = datetime.now(timezone.utc)
    inicio = ahora - timedelta(days=SEARCH_DAYS_BACK)

    inicio_txt = inicio.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    ahora_txt = ahora.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    filtro = (
        f"Collection/Name eq '{COLLECTION}' "
        f"and contains(Name,'{NAME_CONTAINS}') "
        f"and ContentDate/Start ge {inicio_txt} "
        f"and ContentDate/Start le {ahora_txt}"
    )

    params = (
        f"?$filter={quote(filtro, safe='()/,$= ')}"
        f"&$orderby=ContentDate/Start desc"
        f"&$top=1"
    )

    url = CATALOG_URL + params

    print("\nBuscando producto Sentinel-5P SO2 más reciente:")
    print(url)

    r = requests.get(url, timeout=60)
    r.raise_for_status()

    productos = r.json().get("value", [])

    if not productos:
        raise RuntimeError(
            f"No se encontraron productos SO2 en los últimos {SEARCH_DAYS_BACK} días."
        )

    producto = productos[0]

    print("\nProducto encontrado:")
    print("Nombre:", producto["Name"])
    print("Id:", producto["Id"])
    print("Fecha:", producto["ContentDate"]["Start"])

    return producto


def descargar_producto(producto, token):
    RAW_S5P.mkdir(parents=True, exist_ok=True)

    product_id = producto["Id"]
    nombre = producto["Name"]

    if not nombre.endswith(".nc"):
        nombre = nombre + ".nc"

    out_path = RAW_S5P / nombre

    if out_path.exists() and out_path.stat().st_size > 0:
        print("\nEl producto ya existe localmente:")
        print(out_path)
        return out_path

    url = f"{DOWNLOAD_URL}({product_id})/$value"

    headers = {
        "Authorization": f"Bearer {token}"
    }

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    print("\nDescargando producto:")
    print(url)
    print("Salida:", out_path)

    with requests.Session() as session:
        session.headers.update(headers)

        with session.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()

            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    tmp_path.rename(out_path)

    print("\nDescarga completa:")
    print(out_path)

    return out_path


def limpiar_nc_viejos():
    limite = datetime.now().timestamp() - KEEP_DAYS * 24 * 3600

    borrados = 0

    for archivo in RAW_S5P.glob("*.nc"):
        if archivo.stat().st_mtime < limite:
            print("Borrando NC viejo:", archivo)
            archivo.unlink()
            borrados += 1

    print(f"\nLimpieza RAW_S5P terminada. Archivos borrados: {borrados}")


# ============================================================
# LECTURA SENTINEL-5P
# ============================================================

def abrir_grupo_product(ruta_nc: Path):
    return xr.open_dataset(ruta_nc, group="PRODUCT")


def obtener_variables_principales(ds: xr.Dataset):
    lat = ds["latitude"].isel(time=0)
    lon = ds["longitude"].isel(time=0)
    qa = ds["qa_value"].isel(time=0)
    so2 = ds["sulfurdioxide_total_vertical_column"].isel(time=0)

    return lat, lon, qa, so2


def aplicar_filtro_calidad(so2, qa, qa_min=0.5):
    mascara_qa = qa >= qa_min
    so2_filtrado = xr.where(mascara_qa, so2, np.nan)
    return so2_filtrado, mascara_qa


def recortar_region(lat, lon, data, bbox):
    lon_min, lat_min, lon_max, lat_max = bbox

    mascara_roi = (
        (lat >= lat_min)
        & (lat <= lat_max)
        & (lon >= lon_min)
        & (lon <= lon_max)
    )

    data_roi = xr.where(mascara_roi, data, np.nan)

    return data_roi, mascara_roi


def preparar_datos_grafica(lat, lon, so2_roi):
    lon_1d = lon.values.flatten()
    lat_1d = lat.values.flatten()
    so2_1d = so2_roi.values.flatten()

    mask = np.isfinite(so2_1d)

    return lon_1d[mask], lat_1d[mask], so2_1d[mask]


def definir_escala_color(so2_1d):
    if USAR_ESCALA_FIJA:
        return VMIN_FIJO, VMAX_FIJO

    vmax = np.nanpercentile(so2_1d, PERCENTIL_VMAX)

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = np.nanmax(so2_1d)

    return 0.0, vmax


def fecha_desde_nombre_s5p(archivo_nc: Path):
    nombre = archivo_nc.name

    # Ejemplo:
    # S5P_NRTI_L2__SO2____20260326T192229_...
    partes = nombre.split("____")

    if len(partes) > 1:
        fecha = partes[1][:8]
        try:
            return datetime.strptime(fecha, "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            pass

    return datetime.utcnow().strftime("%Y-%m-%d")


# ============================================================
# GRAFICA
# ============================================================

def graficar_so2(
    lat,
    lon,
    so2_roi,
    bbox,
    nombre_volcan,
    lat_volcan,
    lon_volcan,
    archivo_nc,
):
    lon_1d, lat_1d, so2_1d = preparar_datos_grafica(lat, lon, so2_roi)

    if so2_1d.size == 0:
        print(f"No hay datos válidos para graficar en {nombre_volcan}.")
        return

    vmin, vmax = definir_escala_color(so2_1d)

    lon_min, lat_min, lon_max, lat_max = bbox
    fecha = fecha_desde_nombre_s5p(archivo_nc)

    out_dir = OUT_SO2 / fecha / nombre_volcan
    out_dir.mkdir(parents=True, exist_ok=True)

    out_name = f"{nombre_volcan}_{archivo_nc.stem}_SO2.png"
    out_path = out_dir / out_name

    plt.figure(figsize=(10, 8))

    sc = plt.scatter(
        lon_1d,
        lat_1d,
        c=so2_1d,
        s=30,
        marker="s",
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )

    cbar = plt.colorbar(sc)
    cbar.set_label("SO2 total vertical column (mol m$^{-2}$)")

    plt.scatter(
        lon_volcan,
        lat_volcan,
        marker="^",
        s=140,
        color="deepskyblue",
        edgecolor="black",
        label=nombre_volcan,
        zorder=5,
    )

    plt.xlim(lon_min, lon_max)
    plt.ylim(lat_min, lat_max)

    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.title(f"Sentinel-5P SO2 | {nombre_volcan} | {fecha}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("\nPNG generado:")
    print(out_path)


# ============================================================
# PROCESAMIENTO
# ============================================================

def procesar_archivo_s5p(archivo_nc: Path):
    print("\nProcesando archivo:")
    print(archivo_nc)

    ds = abrir_grupo_product(archivo_nc)

    try:
        lat, lon, qa, so2 = obtener_variables_principales(ds)

        print("\n========== RESUMEN ==========")
        print(f"Dimensiones: {dict(ds.dims)}")
        print(f"latitude shape : {lat.shape}")
        print(f"longitude shape: {lon.shape}")
        print(f"qa_value shape : {qa.shape}")
        print(f"so2 shape      : {so2.shape}")

        so2_filtrado, _ = aplicar_filtro_calidad(
            so2,
            qa,
            qa_min=QA_MIN
        )

        for nombre_volcan, info in VOLCANES.items():
            print(f"\nProcesando volcán: {nombre_volcan}")

            bbox = info["bbox"]

            so2_roi, _ = recortar_region(
                lat,
                lon,
                so2_filtrado,
                bbox=bbox
            )

            n_roi = int(np.isfinite(so2_roi.values).sum())

            print(f"Pixeles válidos dentro de ROI: {n_roi}")

            if n_roi == 0:
                print(f"No quedaron pixeles válidos para {nombre_volcan}.")
                continue

            print(
                "SO2 ROI min/max:",
                f"{float(np.nanmin(so2_roi.values)):.6e}",
                "/",
                f"{float(np.nanmax(so2_roi.values)):.6e}",
            )

            graficar_so2(
                lat=lat,
                lon=lon,
                so2_roi=so2_roi,
                bbox=bbox,
                nombre_volcan=nombre_volcan,
                lat_volcan=info["lat"],
                lon_volcan=info["lon"],
                archivo_nc=archivo_nc,
            )

    finally:
        ds.close()


# ============================================================
# MAIN
# ============================================================

def main():
    producto = buscar_producto_so2_mas_reciente()
    token = obtener_token()
    archivo_nc = descargar_producto(producto, token)

    procesar_archivo_s5p(archivo_nc)

    limpiar_nc_viejos()

    print("\nProceso Sentinel-5P SO2 terminado.")
    print("RAW_S5P:", RAW_S5P)
    print("OUT_SO2:", OUT_SO2)


if __name__ == "__main__":
    main()
