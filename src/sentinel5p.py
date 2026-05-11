from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACION
# ============================================================

# Ruta al archivo .nc
archivo_nc = Path("/home/fercz/lanot/data/raw/S5P_NRTI_L2__SO2____20260326T192229_20260326T192729_43788_03_020800_20260326T195822.nc")

# Coordenadas aproximadas del volcan
LAT_VOLCAN = 19.01
LON_VOLCAN = -98.62

# Ventana de recorte alrededor del volcan
DELTA_LAT = 2.0
DELTA_LON = 2.0

# Umbral de calidad recomendado por el manual
QA_MIN = 0.5

# AQUI SE AGREGO: parametros de visualizacion para que se parezca mas al browser
USAR_ESCALA_FIJA = True
VMIN_FIJO = 0.0
VMAX_FIJO = 1.0e-2

# AQUI SE AGREGO: percentil si se quiere escala automatica robusta
PERCENTIL_VMAX = 99.0


# ============================================================
# FUNCIONES
# ============================================================

def abrir_grupo_product(ruta_nc: Path) -> xr.Dataset:
    """
    Abre el grupo PRODUCT del archivo NetCDF Sentinel-5P.
    """
    return xr.open_dataset(ruta_nc, group="PRODUCT")


def obtener_variables_principales(ds: xr.Dataset):
    """
    Extrae las variables principales del producto SO2.
    Se toma el primer indice temporal porque la dimension time suele ser de longitud 1.
    """
    lat = ds["latitude"].isel(time=0)
    lon = ds["longitude"].isel(time=0)
    qa = ds["qa_value"].isel(time=0)
    so2 = ds["sulfurdioxide_total_vertical_column"].isel(time=0)

    return lat, lon, qa, so2


def aplicar_filtro_calidad(so2, qa, qa_min=0.5):
    """
    Conserva solo pixeles con qa_value >= qa_min.
    El manual recomienda ignorar qa_value < 0.5.
    """
    # AQUI SE CAMBIO: se usa >= 0.5 de forma explicita
    mascara_qa = qa >= qa_min
    so2_filtrado = xr.where(mascara_qa, so2, np.nan)
    return so2_filtrado, mascara_qa


def recortar_region(lat, lon, data, lat0, lon0, dlat=2.0, dlon=2.0):
    """
    Aplica un recorte geografico simple mediante una caja lat/lon.
    """
    mascara_roi = (
        (lat >= lat0 - dlat) & (lat <= lat0 + dlat) &
        (lon >= lon0 - dlon) & (lon <= lon0 + dlon)
    )
    data_roi = xr.where(mascara_roi, data, np.nan)
    return data_roi, mascara_roi


def imprimir_resumen(ds, lat, lon, qa, so2):
    """
    Imprime informacion basica para entender el archivo.
    """
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
    print(f"qa min/max  : {float(np.nanmin(qa.values)):.3f} / {float(np.nanmax(qa.values)):.3f}")
    print(f"so2 min/max : {float(np.nanmin(so2.values)):.6e} / {float(np.nanmax(so2.values)):.6e}")


def preparar_datos_grafica(lat, lon, so2_roi):
    """
    Convierte a arreglos 1D y elimina NaNs para graficar.
    """
    # AQUI SE AGREGO: usar solo pixeles validos para no mandar basura al scatter
    lon_1d = lon.values.flatten()
    lat_1d = lat.values.flatten()
    so2_1d = so2_roi.values.flatten()

    mask = np.isfinite(so2_1d)
    return lon_1d[mask], lat_1d[mask], so2_1d[mask]


def definir_escala_color(so2_1d):
    """
    Define vmin y vmax para la grafica.
    """
    # AQUI SE AGREGO: opcion de escala fija o robusta por percentil
    if USAR_ESCALA_FIJA:
        return VMIN_FIJO, VMAX_FIJO

    vmax = np.nanpercentile(so2_1d, PERCENTIL_VMAX)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = np.nanmax(so2_1d)

    vmin = 0.0
    return vmin, vmax


def graficar_so2(lat, lon, so2_roi, lat0, lon0):
    """
    Grafica el SO2 filtrado y recortado.
    Se usa scatter porque el swath TROPOMI no es una malla regular cartesiana.
    """
    lon_1d, lat_1d, so2_1d = preparar_datos_grafica(lat, lon, so2_roi)

    if so2_1d.size == 0:
        print("No hay datos validos para graficar.")
        return

    vmin, vmax = definir_escala_color(so2_1d)

    plt.figure(figsize=(10, 8))

    # AQUI SE CAMBIO: puntos cuadrados y mas grandes para parecerse mas al browser
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
    cbar.set_label("SO2 total vertical column (mol m$^{-2}$)")

    # AQUI SE CAMBIO: nombre correcto del volcan
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

    plt.xlim(lon0 - DELTA_LON, lon0 + DELTA_LON)
    plt.ylim(lat0 - DELTA_LAT, lat0 + DELTA_LAT)

    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.title("Sentinel-5P SO2 filtrado alrededor de Popocatépetl")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    if not archivo_nc.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {archivo_nc}")

    ds = abrir_grupo_product(archivo_nc)
    lat, lon, qa, so2 = obtener_variables_principales(ds)

    imprimir_resumen(ds, lat, lon, qa, so2)

    so2_filtrado, mascara_qa = aplicar_filtro_calidad(so2, qa, qa_min=QA_MIN)
    so2_roi, mascara_roi = recortar_region(
        lat, lon, so2_filtrado,
        lat0=LAT_VOLCAN, lon0=LON_VOLCAN,
        dlat=DELTA_LAT, dlon=DELTA_LON
    )

    print("\n========== CONTEOS ==========")
    print(f"Pixeles totales           : {np.isfinite(so2.values).sum()}")
    print(f"Pixeles con QA valido     : {np.isfinite(so2_filtrado.values).sum()}")
    print(f"Pixeles dentro de la ROI  : {np.isfinite(so2_roi.values).sum()}")

    if np.isfinite(so2_roi.values).sum() == 0:
        print("\nNo quedaron pixeles validos dentro de la region de interes.")
    else:
        print("\n========== ESTADISTICAS ROI ==========")
        print(f"SO2 ROI min/max : {float(np.nanmin(so2_roi.values)):.6e} / {float(np.nanmax(so2_roi.values)):.6e}")
        graficar_so2(lat, lon, so2_roi, LAT_VOLCAN, LON_VOLCAN)

    ds.close()


if __name__ == "__main__":
    main()