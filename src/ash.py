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

# Producto ABI Full Disk usado por este flujo.
PRODUCTOS_GOES = [
    "ABI-L2-CMIPF",
]

PRODUCTO_RGB = "ABI-L2-CMIPF"

# Bandas necesarias para los RGB volcánicos.
BANDAS_DESCARGA = ["C07", "C11", "C13", "C14", "C15"]
BANDAS_RGB = ["C07", "C11", "C13", "C14", "C15"]

# Genera un NAV.nc por volcán/fecha tomando C13 como referencia.
GENERAR_NAV_NC = True
BANDA_NAV = "C13"

# CSV del proyecto.
BASE_DIR = Path(__file__).resolve().parents[1]
EVENTOS_DIR = BASE_DIR / "data" / "eventos"
VOLCANES_CSV = EVENTOS_DIR / "volcanes.csv"
EVENTOS_CSV = EVENTOS_DIR / "eventos.csv"


# ==========================================================
# LECTURA CSV
# ==========================================================

def leer_volcanes_csv(path=VOLCANES_CSV):
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

            datetime.strptime(fecha, "%Y-%m-%d")

            eventos.append({
                "volcan": volcan,
                "fecha": fecha,
                "hora_inicio": hora_inicio,
                "hora_fin": hora_fin,
            })

    return eventos


def evento_tiene_hora(evento):
    return bool(evento.get("hora_inicio") or evento.get("hora_fin"))


def filtrar_eventos(
    eventos,
    volcan=None,
    fecha=None,
    year=None,
    solo_con_hora=False,
    solo_sin_hora=False,
    max_eventos=None,
):
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
        filtrados = [ev for ev in filtrados if ev["fecha"] == fecha]

    if year:
        year = str(year).strip()
        if not year.isdigit() or len(year) != 4:
            raise ValueError("year debe ser algo como 2020")
        filtrados = [
            ev for ev in filtrados
            if ev["fecha"].startswith(year + "-")
        ]

    if solo_con_hora:
        filtrados = [ev for ev in filtrados if evento_tiene_hora(ev)]

    if solo_sin_hora:
        filtrados = [ev for ev in filtrados if not evento_tiene_hora(ev)]

    if max_eventos is not None:
        max_eventos = int(max_eventos)
        if max_eventos <= 0:
            raise ValueError("max_eventos debe ser mayor que 0")
        filtrados = filtrados[:max_eventos]

    return filtrados


def hora_a_entero(hora_txt):
    if not hora_txt:
        return None

    hora_txt = hora_txt.strip()
    if ":" in hora_txt:
        return int(hora_txt.split(":")[0])
    return int(hora_txt)


def horas_evento(evento):
    hi = hora_a_entero(evento.get("hora_inicio", ""))
    hf = hora_a_entero(evento.get("hora_fin", ""))

    if hi is None and hf is None:
        return list(range(24))

    if hi is not None and hf is None:
        hf = hi

    if hi is None and hf is not None:
        hi = hf

    if hf < hi:
        return list(range(hi, 24))

    return list(range(hi, hf + 1))


def eventos_por_volcan_fecha(eventos):
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


def extraer_datetime(path):
    nombre = Path(path).name
    m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})?", nombre)

    if not m:
        return None

    year = int(m.group(1))
    jday = int(m.group(2))
    hour = int(m.group(3))
    minute = int(m.group(4))
    second = int(m.group(5) or 0)

    dt = datetime(year, 1, 1) + timedelta(days=jday - 1)
    return dt.replace(hour=hour, minute=minute, second=second)


def descargar_archivos_evento(s3, bucket, producto, fecha, horas_utc, destino_tmp):
    """Descarga los originales NOAA requeridos para un volcán/fecha en un temporal."""
    year, jday = fecha_a_juliano(fecha)
    destino_tmp.mkdir(parents=True, exist_ok=True)

    descargados = defaultdict(list)

    for hour in horas_utc:
        prefix = f"{producto}/{year}/{jday:03d}/{hour:02d}/"
        print(f"\nBuscando s3://{bucket}/{prefix}")

        try:
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        except Exception as e:
            print(f"Error AWS: {e}")
            continue

        for obj in resp.get("Contents", []):
            key = obj["Key"]
            nombre = Path(key).name
            banda = extraer_banda(nombre)

            if banda not in BANDAS_DESCARGA:
                continue

            out_dir = destino_tmp / banda
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / nombre

            if not out_path.exists():
                print(f"Descargando: {nombre}")
                try:
                    s3.download_file(bucket, key, str(out_path))
                except Exception as e:
                    print(f"Error descarga: {e}")
                    continue

            descargados[banda].append(out_path)

    for banda in descargados:
        descargados[banda] = sorted(set(descargados[banda]))

    return descargados


# ==========================================================
# RECORTE Y COMBINACIÓN DIARIA
# ==========================================================

def recortar_goes_da(da, ds, bbox):
    lon_min, lat_min, lon_max, lat_max = bbox

    proj_attrs = ds["goes_imager_projection"].attrs
    h = proj_attrs["perspective_point_height"]

    crs_goes = CRS.from_cf(proj_attrs)
    crs_geo = CRS.from_epsg(4326)

    transformer = Transformer.from_crs(
        crs_geo,
        crs_goes,
        always_xy=True,
    )

    x1, y1 = transformer.transform(lon_min, lat_min)
    x2, y2 = transformer.transform(lon_max, lat_max)

    x_min = min(x1, x2) / h
    x_max = max(x1, x2) / h
    y_min = min(y1, y2) / h
    y_max = max(y1, y2) / h

    if da.y[0] > da.y[-1]:
        return da.sel(
            x=slice(x_min, x_max),
            y=slice(y_max, y_min),
        )

    return da.sel(
        x=slice(x_min, x_max),
        y=slice(y_min, y_max),
    )


def construir_dataset_banda(archivos, banda, bbox):
    escenas = []
    referencia_attrs = None
    referencia_proj = None

    for archivo in sorted(archivos):
        dt = extraer_datetime(archivo)
        if dt is None:
            continue

        with xr.open_dataset(archivo) as ds:
            if "CMI" not in ds:
                continue

            da = ds["CMI"].astype(np.float32)
            da = recortar_goes_da(da, ds, bbox).load()

            if da.size == 0:
                continue

            da = da.expand_dims(time=[np.datetime64(dt)])
            da.name = banda

            escenas.append(da)

            if referencia_attrs is None:
                referencia_attrs = dict(ds.attrs)
                referencia_proj = dict(ds["goes_imager_projection"].attrs)

    if not escenas:
        return None

    combinado = xr.concat(escenas, dim="time").sortby("time")

    ds_out = combinado.to_dataset(name=banda)
    ds_out["goes_imager_projection"] = xr.DataArray(
        np.int32(0),
        attrs=referencia_proj or {},
    )

    ds_out[banda].attrs["grid_mapping"] = "goes_imager_projection"
    ds_out[banda].attrs["long_name"] = f"GOES ABI {banda} CMI"

    ds_out.attrs.update(referencia_attrs or {})
    ds_out.attrs.update({
        "title": f"GOES ABI {banda} recortado y combinado por fecha",
        "band": banda,
    })

    return ds_out


def guardar_banda_diaria(volcan, fecha, banda, ds_out):
    out_dir = INPUT_BASE / volcan / fecha / banda
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{banda}_{fecha}.nc"

    encoding = {
        banda: {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "dtype": "float32",
            "_FillValue": np.float32(-9999.0),
        }
    }

    ds_out.to_netcdf(
        out_path,
        engine="netcdf4",
        encoding=encoding,
    )

    print(f"Guardado: {out_path}")
    return out_path


def procesar_descarga_evento(s3, volcan, fecha, horas_utc, bbox):
    bucket = obtener_bucket_goes(fecha)
    producto = PRODUCTOS_GOES[0]

    print("\n====================================")
    print(f"VOLCÁN: {volcan}")
    print(f"FECHA:  {fecha}")
    print(f"BUCKET: {bucket}")
    print("====================================")

    tmp_base = INPUT_BASE / ".tmp_goes" / volcan / fecha

    descargados = descargar_archivos_evento(
        s3=s3,
        bucket=bucket,
        producto=producto,
        fecha=fecha,
        horas_utc=horas_utc,
        destino_tmp=tmp_base,
    )

    resultados = {}

    for banda in BANDAS_DESCARGA:
        archivos = descargados.get(banda, [])

        if not archivos:
            print(f"Sin archivos para {banda}")
            continue

        print(f"\nCombinando {banda}: {len(archivos)} escenas")
        ds_out = construir_dataset_banda(archivos, banda, bbox)

        if ds_out is None:
            print(f"No se pudo construir {banda}")
            continue

        try:
            resultados[banda] = guardar_banda_diaria(
                volcan=volcan,
                fecha=fecha,
                banda=banda,
                ds_out=ds_out,
            )
        finally:
            ds_out.close()

    return resultados


# ==========================================================
# NAV.nc POR VOLCÁN / FECHA
# ==========================================================

def calcular_latlon_goes(ds):
    if "goes_imager_projection" not in ds:
        raise ValueError("El NetCDF no contiene goes_imager_projection")
    if "x" not in ds or "y" not in ds:
        raise ValueError("El NetCDF no contiene coordenadas x/y")

    proj_attrs = ds["goes_imager_projection"].attrs
    h = float(proj_attrs["perspective_point_height"])

    crs_goes = CRS.from_cf(proj_attrs)
    crs_geo = CRS.from_epsg(4326)

    transformer = Transformer.from_crs(
        crs_goes,
        crs_geo,
        always_xy=True,
    )

    x = np.asarray(ds["x"].values, dtype=np.float64) * h
    y = np.asarray(ds["y"].values, dtype=np.float64) * h

    xx, yy = np.meshgrid(x, y)
    lon, lat = transformer.transform(xx, yy)

    lat = np.asarray(lat, dtype=np.float32)
    lon = np.asarray(lon, dtype=np.float32)

    invalid = ~np.isfinite(lat) | ~np.isfinite(lon)
    lat[invalid] = np.nan
    lon[invalid] = np.nan

    return lat, lon


def generar_nav_nc(volcan, fecha, archivo_referencia):
    out_dir = INPUT_BASE / volcan / fecha / "NAV"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nav = out_dir / f"GOES_NAV_{fecha}.nc"

    print(f"\nGenerando NAV: {out_nav}")

    with xr.open_dataset(archivo_referencia) as ds:
        lat, lon = calcular_latlon_goes(ds)

        nav_ds = xr.Dataset(
            data_vars={
                "latitude": (
                    ("y", "x"),
                    lat,
                    {
                        "long_name": "latitude",
                        "standard_name": "latitude",
                        "units": "degrees_north",
                    },
                ),
                "longitude": (
                    ("y", "x"),
                    lon,
                    {
                        "long_name": "longitude",
                        "standard_name": "longitude",
                        "units": "degrees_east",
                    },
                ),
                "goes_imager_projection": (
                    (),
                    np.int32(0),
                    dict(ds["goes_imager_projection"].attrs),
                ),
            },
            coords={
                "x": ds["x"].astype(np.float32),
                "y": ds["y"].astype(np.float32),
            },
            attrs={
                "title": "GOES ABI navigation latitude/longitude",
                "volcan": volcan,
                "date": fecha,
                "source_file": Path(archivo_referencia).name,
            },
        )

        encoding = {
            "latitude": {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
                "_FillValue": np.float32(-9999.0),
            },
            "longitude": {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
                "_FillValue": np.float32(-9999.0),
            },
        }

        nav_ds.to_netcdf(
            out_nav,
            engine="netcdf4",
            encoding=encoding,
        )
        nav_ds.close()

    return out_nav


# ==========================================================
# RGB
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
    return np.nan_to_num(data, nan=0.0)


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


def cargar_bandas_diarias(volcan, fecha):
    datasets = {}

    for banda in BANDAS_RGB:
        ruta = INPUT_BASE / volcan / fecha / banda / f"{banda}_{fecha}.nc"
        if not ruta.exists():
            for ds in datasets.values():
                ds.close()
            return None
        datasets[banda] = xr.open_dataset(ruta)

    return datasets


def guardar_rgb(
    rgb,
    out_path,
    titulo,
    bbox,
    lon_volcan,
    lat_volcan,
    nombre_volcan,
):
    lon_min, lat_min, lon_max, lat_max = bbox

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(
        rgb,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin="upper",
    )

    ax.scatter(
        lon_volcan,
        lat_volcan,
        s=45,
        marker="^",
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
    )

    ax.text(
        lon_volcan + 0.08,
        lat_volcan + 0.08,
        nombre_volcan,
        fontsize=8,
        color="white",
        bbox=dict(facecolor="black", alpha=0.45, edgecolor="none"),
        zorder=6,
    )

    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def procesar_rgb_evento(volcan, fecha, info, horas_permitidas):
    datasets = cargar_bandas_diarias(volcan, fecha)

    if datasets is None:
        print(f"Faltan bandas diarias para RGB: {volcan} {fecha}")
        return

    try:
        tiempos = datasets["C13"]["time"].values

        for t in tiempos:
            dt = np.datetime64(t).astype("datetime64[s]").astype(datetime)

            if dt.hour not in horas_permitidas:
                continue

            timestamp = dt.strftime("%Y-%m-%d_%H%M")
            print(f"Procesando RGB {volcan} | {timestamp}")

            valores = {}
            for banda in BANDAS_RGB:
                da = datasets[banda][banda]
                idx = np.argmin(np.abs(da["time"].values - np.datetime64(dt)))
                valores[banda] = np.asarray(da.isel(time=idx).values, dtype=float)

            rgbs = crear_rgbs(
                valores["C07"],
                valores["C11"],
                valores["C13"],
                valores["C14"],
                valores["C15"],
            )

            for nombre_rgb, rgb in rgbs.items():
                out_dir = OUTPUT_BASE / volcan / fecha / nombre_rgb
                out_dir.mkdir(parents=True, exist_ok=True)

                out_path = out_dir / f"{volcan}_{timestamp}_{nombre_rgb}.png"
                titulo = f"{volcan} | {timestamp} UTC | {nombre_rgb}"

                guardar_rgb(
                    rgb=rgb,
                    out_path=out_path,
                    titulo=titulo,
                    bbox=info["bbox"],
                    lon_volcan=info["lon"],
                    lat_volcan=info["lat"],
                    nombre_volcan=volcan,
                )
    finally:
        for ds in datasets.values():
            ds.close()


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
    max_eventos=None,
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
        max_eventos=max_eventos,
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

    eventos_vf = eventos_por_volcan_fecha(eventos)

    s3 = boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED),
    )

    for (nombre_volcan, fecha_evento), horas in sorted(eventos_vf.items()):
        info = volcanes[nombre_volcan]

        if modo in {"todo", "descarga"}:
            resultados = procesar_descarga_evento(
                s3=s3,
                volcan=nombre_volcan,
                fecha=fecha_evento,
                horas_utc=horas,
                bbox=info["bbox"],
            )

            if GENERAR_NAV_NC and BANDA_NAV in resultados:
                generar_nav_nc(
                    volcan=nombre_volcan,
                    fecha=fecha_evento,
                    archivo_referencia=resultados[BANDA_NAV],
                )

        if modo in {"todo", "rgb"}:
            procesar_rgb_evento(
                volcan=nombre_volcan,
                fecha=fecha_evento,
                info=info,
                horas_permitidas=horas,
            )

    print("\nProceso GOES terminado.")
    print("RAW_GOES:", INPUT_BASE)
    print("OUT_RGB_ASH:", OUTPUT_BASE)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline GOES para descarga, combinación diaria y RGB de ceniza."
    )

    parser.add_argument(
        "--modo",
        choices=["todo", "descarga", "rgb"],
        default="todo",
        help="todo=descarga+rgb, descarga=solo descarga, rgb=solo procesa RGB",
    )
    parser.add_argument("--volcan", default=None)
    parser.add_argument("--fecha", default=None, help="YYYY-MM-DD")
    parser.add_argument("--year", default=None)
    parser.add_argument("--solo-con-hora", action="store_true")
    parser.add_argument("--solo-sin-hora", action="store_true")
    parser.add_argument("--max-eventos", type=int, default=None)

    args = parser.parse_args()

    run(
        modo=args.modo,
        volcan=args.volcan,
        fecha=args.fecha,
        year=args.year,
        solo_con_hora=args.solo_con_hora,
        solo_sin_hora=args.solo_sin_hora,
        max_eventos=args.max_eventos,
    )


if __name__ == "__main__":
    main()
