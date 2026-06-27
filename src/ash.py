from pathlib import Path
import re
import csv
from datetime import datetime, timedelta
from collections import defaultdict

import boto3
from botocore import UNSIGNED
from botocore.config import Config

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer

from config import RAW_GOES, OUT_RGB_ASH


# ==========================================================
# CONFIGURACIÓN GENERAL
# ==========================================================

INPUT_BASE = RAW_GOES
OUTPUT_BASE = OUT_RGB_ASH

# Bucket GOES se decide automaticamente por fecha del evento

# Productos ABI Full Disk a descargar
PRODUCTOS_GOES = [
    "ABI-L1b-RadF",
    "ABI-L2-ACTPF",
    "ABI-L2-CMIPF",
]

# Producto que se usará para generar los RGB
PRODUCTO_RGB = "ABI-L2-CMIPF"

# Todas las bandas ABI para descarga
BANDAS_DESCARGA = [
    "C01", "C02", "C03", "C04",
    "C05", "C06", "C07", "C08",
    "C09", "C10", "C11", "C12",
    "C13", "C14", "C15", "C16"
]

# Bandas necesarias para RGB volcánicos
BANDAS_RGB = ["C07", "C11", "C13", "C14", "C15"]

# CSV del proyecto
BASE_DIR = Path(__file__).resolve().parents[1]
EVENTOS_DIR = BASE_DIR / "data" / "eventos"
VOLCANES_CSV = EVENTOS_DIR / "volcanes.csv"
EVENTOS_CSV = EVENTOS_DIR / "eventos.csv"


# ==========================================================
# LECTURA CSV
# ==========================================================

def leer_volcanes_csv(path=VOLCANES_CSV):
    """
    Lee data/eventos/volcanes.csv

    Formato:
    volcan,lat,lon,lon_min,lat_min,lon_max,lat_max
    """

    if not Path(path).exists():
        raise FileNotFoundError(
            f"No existe {path}. Crea data/eventos/volcanes.csv"
        )

    volcanes = {}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            nombre = row["volcan"].strip()

            volcanes[nombre] = {
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "bbox": [
                    float(row["lon_min"]),
                    float(row["lat_min"]),
                    float(row["lon_max"]),
                    float(row["lat_max"]),
                ],
            }

    return volcanes


def leer_eventos_csv(path=EVENTOS_CSV):
    """
    Lee data/eventos/eventos.csv

    Formato:
    volcan,fecha,hora_inicio,hora_fin

    - fecha en YYYY-MM-DD
    - hora_inicio y hora_fin opcionales.
    - si no hay horas, se descarga todo el día.
    """

    if not Path(path).exists():
        raise FileNotFoundError(
            f"No existe {path}. Crea data/eventos/eventos.csv"
        )

    eventos = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            volcan = row["volcan"].strip()
            fecha = row["fecha"].strip()
            hora_inicio = (row.get("hora_inicio") or "").strip()
            hora_fin = (row.get("hora_fin") or "").strip()

            if not volcan or not fecha:
                continue

            # valida formato de fecha
            datetime.strptime(fecha, "%Y-%m-%d")

            eventos.append({
                "volcan": volcan,
                "fecha": fecha,
                "hora_inicio": hora_inicio,
                "hora_fin": hora_fin,
            })

    return eventos



def evento_tiene_hora(evento):
    """
    True si el evento tiene al menos una hora definida.
    """

    return bool(
        evento.get("hora_inicio")
        or evento.get("hora_fin")
    )


def filtrar_eventos(
    eventos,
    volcan=None,
    fecha=None,
    year=None,
    solo_con_hora=False,
    solo_sin_hora=False,
    max_eventos=None
):
    """
    Filtra la lista de eventos sin modificar el CSV original.

    - volcan: nombre del volcan como aparece en eventos.csv.
    - fecha: YYYY-MM-DD.
    - year: anio de cuatro digitos.
    - solo_con_hora: usa eventos con hora_inicio/hora_fin.
    - solo_sin_hora: usa eventos sin horas, o sea dias completos.
    - max_eventos: limita la cantidad de filas de eventos procesadas.
    """

    if solo_con_hora and solo_sin_hora:
        raise ValueError(
            "No uses --solo-con-hora y --solo-sin-hora al mismo tiempo"
        )

    filtrados = list(eventos)

    if volcan:
        volcan = volcan.strip().lower()
        filtrados = [
            ev for ev in filtrados
            if ev["volcan"].strip().lower() == volcan
        ]

    if fecha:
        fecha = fecha.strip()
        datetime.strptime(fecha, "%Y-%m-%d")
        filtrados = [
            ev for ev in filtrados
            if ev["fecha"] == fecha
        ]

    if year:
        year = str(year).strip()
        if not year.isdigit() or len(year) != 4:
            raise ValueError("year debe ser algo como 2020")

        filtrados = [
            ev for ev in filtrados
            if ev["fecha"].startswith(year + "-")
        ]

    if solo_con_hora:
        filtrados = [
            ev for ev in filtrados
            if evento_tiene_hora(ev)
        ]

    if solo_sin_hora:
        filtrados = [
            ev for ev in filtrados
            if not evento_tiene_hora(ev)
        ]

    if max_eventos is not None:
        max_eventos = int(max_eventos)
        if max_eventos <= 0:
            raise ValueError("max_eventos debe ser mayor que 0")

        filtrados = filtrados[:max_eventos]

    return filtrados

def hora_a_entero(hora_txt):
    """
    Convierte '13:46' -> 13, '13' -> 13.
    """

    if not hora_txt:
        return None

    hora_txt = hora_txt.strip()

    if ":" in hora_txt:
        return int(hora_txt.split(":")[0])

    return int(hora_txt)


def horas_evento(evento):
    """
    Devuelve las horas UTC a descargar para un evento.

    Si no hay hora_inicio/hora_fin: descarga 0-23.
    Si hay rango: descarga horas enteras inclusivas.
    Si hora_fin < hora_inicio, asume que cruza medianoche y
    para esa fecha descarga desde hora_inicio hasta 23.
    La continuación debe ponerse como otra fila del día siguiente.
    """

    hi = hora_a_entero(evento.get("hora_inicio", ""))
    hf = hora_a_entero(evento.get("hora_fin", ""))

    if hi is None and hf is None:
        return list(range(0, 24))

    if hi is not None and hf is None:
        hf = hi

    if hi is None and hf is not None:
        hi = hf

    if hf < hi:
        return list(range(hi, 24))

    return list(range(hi, hf + 1))


def eventos_por_fecha(eventos):
    """
    Regresa:
    {
      '2023-09-23': {8,9,10,11,12,13},
      ...
    }
    """

    salida = defaultdict(set)

    for ev in eventos:
        for h in horas_evento(ev):
            salida[ev["fecha"]].add(h)

    return {
        fecha: sorted(horas)
        for fecha, horas in salida.items()
    }


def eventos_por_volcan_fecha(eventos):
    """
    Regresa:
    {
      ('Villarrica','2023-09-23'): {8,9,10,11,12,13},
      ...
    }
    """

    salida = defaultdict(set)

    for ev in eventos:
        key = (ev["volcan"], ev["fecha"])

        for h in horas_evento(ev):
            salida[key].add(h)

    return {
        key: sorted(horas)
        for key, horas in salida.items()
    }


# ==========================================================
# AWS S3
# ==========================================================

def fecha_a_juliano(fecha):

    dt = datetime.strptime(fecha, "%Y-%m-%d")
    return dt.year, dt.timetuple().tm_yday


def obtener_bucket_goes(fecha):
    """
    Decide automaticamente que bucket usar segun la fecha del evento.

    Para eventos historicos se usa GOES-16.
    Para fechas posteriores al relevo operacional de GOES-East se usa GOES-19.
    """

    fecha_dt = datetime.strptime(fecha, "%Y-%m-%d")
    cambio_goes19 = datetime(2025, 4, 7)

    if fecha_dt < cambio_goes19:
        return "noaa-goes16"

    return "noaa-goes19"


def extraer_banda(path):

    nombre = Path(path).name

    m = re.search(r"C(0[1-9]|1[0-6])", nombre)

    if m:
        return "C" + m.group(1)

    return None


def descargar_goes_eventos(eventos):

    fechas_horas = eventos_por_fecha(eventos)

    print("\n==============================")
    print("CONFIGURACIÓN DESCARGA GOES")
    print("==============================")
    print("Bucket: automatico por fecha")
    print("Productos:", PRODUCTOS_GOES)
    print("Fechas:", list(fechas_horas.keys()))
    print("Bandas descarga:", BANDAS_DESCARGA)
    print("Input:", INPUT_BASE)
    print("==============================\n")

    s3 = boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED)
    )

    for fecha, horas_utc in fechas_horas.items():

        year, jday = fecha_a_juliano(fecha)
        bucket_goes = obtener_bucket_goes(fecha)

        print(f"\nFecha: {fecha} -> {bucket_goes}")

        for producto in PRODUCTOS_GOES:

            for hour in horas_utc:

                prefix = f"{producto}/{year}/{jday:03d}/{hour:02d}/"

                print(f"\nBuscando:")
                print(f"s3://{bucket_goes}/{prefix}")

                try:

                    resp = s3.list_objects_v2(
                        Bucket=bucket_goes,
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

                    nombre = Path(key).name
                    banda = extraer_banda(nombre)

                    # ======================================================
                    # ACTPF no usa bandas C01-C16
                    # ======================================================

                    if producto == "ABI-L2-ACTPF":

                        out_dir = (
                            INPUT_BASE
                            / producto
                            / fecha
                        )

                    else:

                        if banda not in BANDAS_DESCARGA:
                            continue

                        # RAW_GOES / producto / fecha / banda / archivo.nc
                        # La hora ya viene en el nombre del archivo GOES.
                        out_dir = (
                            INPUT_BASE
                            / producto
                            / fecha
                            / banda
                        )

                    out_dir.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    out_path = out_dir / nombre

                    if out_path.exists():
                        print(f"Ya existe: {nombre}")
                        continue

                    print(f"Descargando: {nombre}")

                    try:

                        s3.download_file(
                            bucket_goes,
                            key,
                            str(out_path)
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


def extraer_timestamp(path):

    nombre = Path(path).name

    m = re.search(
        r"_s(\d{4})(\d{3})(\d{2})(\d{2})",
        nombre
    )

    if m:

        year = int(m.group(1))
        jday = int(m.group(2))
        hour = int(m.group(3))
        minute = int(m.group(4))

        dt = datetime(year, 1, 1) + timedelta(days=jday - 1)

        dt = dt.replace(
            hour=hour,
            minute=minute,
            second=0
        )

        return dt.strftime("%Y-%m-%d_%H%M")

    return "timestamp_desconocido"


def buscar_y_agrupar_archivos(input_base):

    archivos = list(
        input_base.rglob("*.nc")
    )

    print("\nArchivos .nc encontrados:", len(archivos))

    grupos = {}

    for archivo in archivos:

        banda = extraer_banda(archivo)

        if banda not in BANDAS_RGB:
            continue

        timestamp = extraer_timestamp(archivo)

        if timestamp not in grupos:
            grupos[timestamp] = {}

        grupos[timestamp][banda] = archivo

    grupos_completos = {

        t: bandas
        for t, bandas in grupos.items()
        if all(b in bandas for b in BANDAS_RGB)

    }

    return grupos_completos


def timestamp_a_fecha_hora(timestamp):
    """
    '2023-09-23_0841' -> ('2023-09-23', 8)
    """

    fecha, hhmm = timestamp.split("_")
    hora = int(hhmm[:2])

    return fecha, hora


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

    x1, y1 = transformer.transform(
        lon_min,
        lat_min
    )

    x2, y2 = transformer.transform(
        lon_max,
        lat_max
    )

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
        raise ValueError(
            f"No hay variable CMI en: {archivo}"
        )

    da = ds["CMI"].astype(float)

    da_crop = recortar_goes_da(
        da,
        ds,
        bbox
    )

    valores = da_crop.compute().values

    ds.close()

    return valores


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

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.imshow(
        rgb,
        extent=[
            lon_min,
            lon_max,
            lat_min,
            lat_max
        ],
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

    ax.set_title(
        titulo,
        fontsize=11
    )

    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")

    plt.tight_layout()

    plt.savefig(
        out_path,
        dpi=200
    )

    plt.close()


def procesar_rgb(volcanes, eventos):

    eventos_vf = eventos_por_volcan_fecha(eventos)

    grupos = buscar_y_agrupar_archivos(
        INPUT_BASE / PRODUCTO_RGB
    )

    print(
        f"\nEscenas completas encontradas: {len(grupos)}"
    )

    for timestamp, archivos_bandas in sorted(grupos.items()):

        if "-" not in timestamp:
            continue

        fecha_timestamp, hora_timestamp = timestamp_a_fecha_hora(timestamp)

        for nombre_volcan, info in volcanes.items():

            key = (nombre_volcan, fecha_timestamp)

            if key not in eventos_vf:
                continue

            if hora_timestamp not in eventos_vf[key]:
                continue

            print(
                f"\nProcesando {nombre_volcan} | {timestamp}"
            )

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

            fecha_out = fecha_timestamp

            for nombre_rgb, rgb in rgbs.items():

                out_dir = (
                    OUTPUT_BASE
                    / fecha_out
                    / nombre_volcan
                    / nombre_rgb
                )

                out_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                titulo = (
                    f"{nombre_volcan} | "
                    f"{timestamp} UTC | "
                    f"{nombre_rgb}"
                )

                out_name = (
                    f"{nombre_volcan}_"
                    f"{timestamp}_"
                    f"{nombre_rgb}.png"
                )

                out_path = out_dir / out_name

                guardar_rgb(
                    rgb=rgb,
                    out_path=out_path,
                    titulo=titulo,
                    bbox=bbox,
                    lon_volcan=info["lon"],
                    lat_volcan=info["lat"],
                    nombre_volcan=nombre_volcan
                )


# ==========================================================
# RUN
# ==========================================================

def run(
    modo="todo",
    volcan=None,
    fecha=None,
    year=None,
    solo_con_hora=False,
    solo_sin_hora=False,
    max_eventos=None
):

    modo = modo.lower().strip()

    if modo not in {"todo", "descarga", "rgb"}:
        raise ValueError("modo debe ser: todo, descarga o rgb")

    volcanes = leer_volcanes_csv()
    eventos = leer_eventos_csv()
    total_eventos_csv = len(eventos)

    eventos = filtrar_eventos(
        eventos,
        volcan=volcan,
        fecha=fecha,
        year=year,
        solo_con_hora=solo_con_hora,
        solo_sin_hora=solo_sin_hora,
        max_eventos=max_eventos
    )

    print("\nVolcanes cargados:", len(volcanes))
    print("Eventos en CSV:", total_eventos_csv)
    print("Eventos a procesar:", len(eventos))

    if not eventos:
        print("\nNo hay eventos para procesar con esos filtros.")
        return

    volcanes_eventos = sorted(set(ev["volcan"] for ev in eventos))
    faltantes = [v for v in volcanes_eventos if v not in volcanes]

    if faltantes:
        raise ValueError(
            "Hay volcanes en eventos.csv que no existen en volcanes.csv: "
            + ", ".join(faltantes)
        )

    print("Volcanes en esta corrida:", ", ".join(volcanes_eventos))

    fechas_horas = eventos_por_fecha(eventos)
    horas_total = sum(len(horas) for horas in fechas_horas.values())
    print("Fechas a descargar/procesar:", len(fechas_horas))
    print("Horas UTC unicas a revisar:", horas_total)

    if modo in {"todo", "descarga"}:
        descargar_goes_eventos(eventos)

    if modo in {"todo", "rgb"}:
        procesar_rgb(volcanes, eventos)

    print("\nProceso GOES terminado.")
    print("RAW_GOES:", INPUT_BASE)
    print("OUT_RGB_ASH:", OUTPUT_BASE)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline GOES para descarga y RGB de ceniza."
    )

    parser.add_argument(
        "--modo",
        choices=["todo", "descarga", "rgb"],
        default="todo",
        help="todo=descarga+rgb, descarga=solo descarga, rgb=solo procesa RGB"
    )

    parser.add_argument(
        "--volcan",
        default=None,
        help="Procesa solo un volcan, con el nombre usado en eventos.csv"
    )

    parser.add_argument(
        "--fecha",
        default=None,
        help="Procesa solo una fecha exacta: YYYY-MM-DD"
    )

    parser.add_argument(
        "--year",
        default=None,
        help="Procesa solo eventos de un anio, por ejemplo 2020"
    )

    parser.add_argument(
        "--solo-con-hora",
        action="store_true",
        help="Procesa solo eventos con hora_inicio u hora_fin"
    )

    parser.add_argument(
        "--solo-sin-hora",
        action="store_true",
        help="Procesa solo eventos sin horas definidas, o sea dias completos"
    )

    parser.add_argument(
        "--max-eventos",
        type=int,
        default=None,
        help="Limita la cantidad de filas de eventos.csv a procesar"
    )

    args = parser.parse_args()

    run(
        modo=args.modo,
        volcan=args.volcan,
        fecha=args.fecha,
        year=args.year,
        solo_con_hora=args.solo_con_hora,
        solo_sin_hora=args.solo_sin_hora,
        max_eventos=args.max_eventos
    )


if __name__ == "__main__":
    main()
