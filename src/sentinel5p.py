from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from config import RAW_S5P, OUT_SO2


# ============================================================
# CONFIGURACION
# ============================================================

# Coordenadas aproximadas del volcan
LAT_VOLCAN = 19.01
LON_VOLCAN = -98.62

# Ventana de recorte alrededor del volcan
DELTA_LAT = 2.0
DELTA_LON = 2.0

# Umbral de calidad recomendado
QA_MIN = 0.5

# Escala visual
USAR_ESCALA_FIJA = True
VMIN_FIJO = 0.0
VMAX_FIJO = 1.0e-2

# Percentil robusto
PERCENTIL_VMAX = 99.0


# ============================================================
# BUSQUEDA AUTOMATICA DE ARCHIVOS
# ============================================================

def buscar_archivo_mas_reciente():

    archivos = sorted(
        RAW_S5P.glob("*.nc")
    )

    if len(archivos) == 0:
        raise FileNotFoundError(
            f"No se encontraron archivos .nc en: {RAW_S5P}"
        )

    archivo = max(
        archivos,
        key=lambda p: p.stat().st_mtime
    )

    print("\nArchivo Sentinel-5P encontrado:")
    print(archivo)

    return archivo


# ============================================================
# FUNCIONES
# ============================================================

def abrir_grupo_product(ruta_nc: Path):

    return xr.open_dataset(
        ruta_nc,
        group="PRODUCT"
    )


def obtener_variables_principales(ds: xr.Dataset):

    lat = ds["latitude"].isel(time=0)

    lon = ds["longitude"].isel(time=0)

    qa = ds["qa_value"].isel(time=0)

    so2 = ds[
        "sulfurdioxide_total_vertical_column"
    ].isel(time=0)

    return lat, lon, qa, so2


def aplicar_filtro_calidad(
    so2,
    qa,
    qa_min=0.5
):

    mascara_qa = qa >= qa_min

    so2_filtrado = xr.where(
        mascara_qa,
        so2,
        np.nan
    )

    return so2_filtrado, mascara_qa


def recortar_region(
    lat,
    lon,
    data,
    lat0,
    lon0,
    dlat=2.0,
    dlon=2.0
):

    mascara_roi = (
        (lat >= lat0 - dlat)
        & (lat <= lat0 + dlat)
        & (lon >= lon0 - dlon)
        & (lon <= lon0 + dlon)
    )

    data_roi = xr.where(
        mascara_roi,
        data,
        np.nan
    )

    return data_roi, mascara_roi


def imprimir_resumen(
    ds,
    lat,
    lon,
    qa,
    so2
):

    print("\n========== RESUMEN DEL ARCHIVO ==========")

    print(f"Dimensiones: {dict(ds.dims)}")

    print("\nVariables disponibles en PRODUCT:")

    for nombre in ds.data_vars:
        print(f" - {nombre}")

    print("\n========== VARIABLES PRINCIPALES ==========")

    print(f"latitude shape : {lat.shape}")
    print(f"longitude shape: {lon.shape}")
    print(f"qa_value shape : {qa.shape}")
    print(f"so2 shape      : {so2.shape}")

    print("\n========== ESTADISTICAS CRUDAS ==========")

    print(
        f"qa min/max  : "
        f"{float(np.nanmin(qa.values)):.3f} / "
        f"{float(np.nanmax(qa.values)):.3f}"
    )

    print(
        f"so2 min/max : "
        f"{float(np.nanmin(so2.values)):.6e} / "
        f"{float(np.nanmax(so2.values)):.6e}"
    )


def preparar_datos_grafica(
    lat,
    lon,
    so2_roi
):

    lon_1d = lon.values.flatten()

    lat_1d = lat.values.flatten()

    so2_1d = so2_roi.values.flatten()

    mask = np.isfinite(so2_1d)

    return (
        lon_1d[mask],
        lat_1d[mask],
        so2_1d[mask]
    )


def definir_escala_color(so2_1d):

    if USAR_ESCALA_FIJA:
        return VMIN_FIJO, VMAX_FIJO

    vmax = np.nanpercentile(
        so2_1d,
        PERCENTIL_VMAX
    )

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = np.nanmax(so2_1d)

    vmin = 0.0

    return vmin, vmax


def construir_output_path(archivo_nc: Path):

    OUT_SO2.mkdir(
        parents=True,
        exist_ok=True
    )

    nombre_png = (
        archivo_nc.stem + "_SO2.png"
    )

    return OUT_SO2 / nombre_png


def graficar_so2(
    lat,
    lon,
    so2_roi,
    lat0,
    lon0,
    out_path
):

    lon_1d, lat_1d, so2_1d = preparar_datos_grafica(
        lat,
        lon,
        so2_roi
    )

    if so2_1d.size == 0:

        print(
            "No hay datos validos para graficar."
        )

        return

    vmin, vmax = definir_escala_color(
        so2_1d
    )

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
        linewidths=0
    )

    cbar = plt.colorbar(sc)

    cbar.set_label(
        "SO2 total vertical column "
        "(mol m$^{-2}$)"
    )

    plt.scatter(
        lon0,
        lat0,
        marker="^",
        s=140,
        color="deepskyblue",
        edgecolor="black",
        label="Popocatépetl",
        zorder=5
    )

    plt.xlim(
        lon0 - DELTA_LON,
        lon0 + DELTA_LON
    )

    plt.ylim(
        lat0 - DELTA_LAT,
        lat0 + DELTA_LAT
    )

    plt.xlabel("Longitud")

    plt.ylabel("Latitud")

    plt.title(
        "Sentinel-5P SO2 filtrado "
        "alrededor de Popocatépetl"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        out_path,
        dpi=200
    )

    plt.close()

    print("\nPNG generado:")
    print(out_path)


# ============================================================
# MAIN
# ============================================================

def main():

    archivo_nc = buscar_archivo_mas_reciente()

    ds = abrir_grupo_product(
        archivo_nc
    )

    lat, lon, qa, so2 = obtener_variables_principales(
        ds
    )

    imprimir_resumen(
        ds,
        lat,
        lon,
        qa,
        so2
    )

    so2_filtrado, mascara_qa = aplicar_filtro_calidad(
        so2,
        qa,
        qa_min=QA_MIN
    )

    so2_roi, mascara_roi = recortar_region(
        lat,
        lon,
        so2_filtrado,
        lat0=LAT_VOLCAN,
        lon0=LON_VOLCAN,
        dlat=DELTA_LAT,
        dlon=DELTA_LON
    )

    print("\n========== CONTEOS ==========")

    print(
        f"Pixeles totales           : "
        f"{np.isfinite(so2.values).sum()}"
    )

    print(
        f"Pixeles con QA valido     : "
        f"{np.isfinite(so2_filtrado.values).sum()}"
    )

    print(
        f"Pixeles dentro de la ROI  : "
        f"{np.isfinite(so2_roi.values).sum()}"
    )

    if np.isfinite(so2_roi.values).sum() == 0:

        print(
            "\nNo quedaron pixeles validos "
            "dentro de la region de interes."
        )

    else:

        print("\n========== ESTADISTICAS ROI ==========")

        print(
            f"SO2 ROI min/max : "
            f"{float(np.nanmin(so2_roi.values)):.6e} / "
            f"{float(np.nanmax(so2_roi.values)):.6e}"
        )

        out_path = construir_output_path(
            archivo_nc
        )

        graficar_so2(
            lat,
            lon,
            so2_roi,
            LAT_VOLCAN,
            LON_VOLCAN,
            out_path
        )

    ds.close()


if __name__ == "__main__":
    main()
