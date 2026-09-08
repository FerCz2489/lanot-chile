from pathlib import Path
import re
import csv
import shutil
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

# Productos GOES que deben conservarse para cada evento.
# RADF y CMIPF existen por canal C01-C16.
# ACTPF es un producto L2 derivado y NO está dividido por canal.
PRODUCTOS_GOES = [
    "ABI-L1b-RadF",
    "ABI-L2-ACTPF",
    "ABI-L2-CMIPF",
]

BANDAS_ABI = [f"C{i:02d}" for i in range(1, 17)]

# Los RGB de ceniza se construyen a partir de CMIPF.
PRODUCTO_RGB = "ABI-L2-CMIPF"
BANDAS_RGB = ["C07", "C11", "C13", "C14", "C15"]

# Genera un NAV.nc por volcán/fecha tomando C13 de CMIPF como referencia.
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
        # Se mantiene el comportamiento histórico: desde hora_inicio hasta 23 UTC.
        return list(range(hi, 24))

    return list(range(hi, hf + 1))


def eventos_por_volcan_fecha(eventos):
    salida = defaultdict(set)

    for ev in eventos:
        key = (ev["volcan"], ev["fecha"])
        for h in horas_evento(ev):
            salida[key].add(h)

    return {key: sorted(horas) for key, horas in salida.items()}


# ==========================================================
# AWS S3 / NOMBRES NOAA
# ==========================================================

def fecha_a_juliano(fecha):
    dt = datetime.strptime(fecha, "%Y-%m-%d")
    return dt.year, dt.timetuple().tm_yday


def obtener_bucket_goes(fecha):
    """
    Decide automáticamente qué bucket usar según la fecha del evento.

    Para eventos históricos se usa GOES-16.
    Para fechas posteriores al relevo operacional de GOES-East se usa GOES-19.
    """
    fecha_dt = datetime.strptime(fecha, "%Y-%m-%d")
    cambio_goes19 = datetime(2025, 4, 7)

    if fecha_dt < cambio_goes19:
        return "noaa-goes16"

    return "noaa-goes19"


def producto_tiene_bandas(producto):
    return producto in {"ABI-L1b-RadF", "ABI-L2-CMIPF"}


def variable_principal(producto):
    if producto == "ABI-L1b-RadF":
        return "Rad"
    if producto == "ABI-L2-CMIPF":
        return "CMI"
    return None


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


def listar_objetos_prefix(s3, bucket, prefix):
    """Itera sobre TODOS los objetos de un prefix, incluyendo paginación S3."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj


def descargar_archivos_producto(
    s3,
    bucket,
    producto,
    fecha,
    horas_utc,
    destino_tmp,
):
    """
    Descarga originales NOAA para un producto.

    RADF/CMIPF: conserva C01-C16.
    ACTPF: conserva todos los archivos del producto, ya que no está dividido por canal.
    """
    year, jday = fecha_a_juliano(fecha)
    destino_tmp = Path(destino_tmp)
    destino_tmp.mkdir(parents=True, exist_ok=True)

    if producto_tiene_bandas(producto):
        descargados = defaultdict(list)
    else:
        descargados = []

    for hour in horas_utc:
        prefix = f"{producto}/{year}/{jday:03d}/{hour:02d}/"
        print(f"\nBuscando s3://{bucket}/{prefix}")

        try:
            objetos = list(listar_objetos_prefix(s3, bucket, prefix))
        except Exception as e:
            print(f"Error AWS en {producto}: {e}")
            continue

        for obj in objetos:
            key = obj["Key"]
            nombre = Path(key).name

            if producto_tiene_bandas(producto):
                banda = extraer_banda(nombre)
                if banda not in BANDAS_ABI:
                    continue
                out_dir = destino_tmp / producto / banda
            else:
                banda = None
                out_dir = destino_tmp / producto

            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / nombre

            if not out_path.exists():
                print(f"Descargando: {nombre}")
                try:
                    s3.download_file(bucket, key, str(out_path))
                except Exception as e:
                    print(f"Error descarga {nombre}: {e}")
                    continue

            if banda is None:
                descargados.append(out_path)
            else:
                descargados[banda].append(out_path)

    if producto_tiene_bandas(producto):
        for banda in descargados:
            descargados[banda] = sorted(set(descargados[banda]))
    else:
        descargados = sorted(set(descargados))

    return descargados


# ==========================================================
# RECORTE GEOESTACIONARIO
# ==========================================================

def limites_xy_goes(ds, bbox):
    lon_min, lat_min, lon_max, lat_max = bbox

    if "goes_imager_projection" not in ds:
        raise ValueError("El archivo no contiene goes_imager_projection")

    proj_attrs = ds["goes_imager_projection"].attrs
    h = float(proj_attrs["perspective_point_height"])

    crs_goes = CRS.from_cf(proj_attrs)
    crs_geo = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs_geo, crs_goes, always_xy=True)

    # Transformar las cuatro esquinas, no sólo una diagonal.
    esquinas = [
        (lon_min, lat_min),
        (lon_min, lat_max),
        (lon_max, lat_min),
        (lon_max, lat_max),
    ]
    xy = [transformer.transform(lon, lat) for lon, lat in esquinas]
    xs = [p[0] / h for p in xy]
    ys = [p[1] / h for p in xy]

    return min(xs), max(xs), min(ys), max(ys)


def recortar_goes_da(da, ds, bbox):
    x_min, x_max, y_min, y_max = limites_xy_goes(ds, bbox)

    if "x" not in da.dims or "y" not in da.dims:
        return da

    if da.y[0] > da.y[-1]:
        return da.sel(x=slice(x_min, x_max), y=slice(y_max, y_min))

    return da.sel(x=slice(x_min, x_max), y=slice(y_min, y_max))


def recortar_dataset_goes(ds, bbox):
    """Recorta un Dataset conservando sus variables; útil para ACTPF."""
    x_min, x_max, y_min, y_max = limites_xy_goes(ds, bbox)

    if "x" not in ds.coords or "y" not in ds.coords:
        return ds

    if ds["y"][0] > ds["y"][-1]:
        return ds.sel(x=slice(x_min, x_max), y=slice(y_max, y_min))

    return ds.sel(x=slice(x_min, x_max), y=slice(y_min, y_max))


# ==========================================================
# PRODUCTOS POR CANAL: RADF / CMIPF
# ==========================================================

def construir_dataset_banda(archivos, producto, banda, bbox):
    """
    Combina escenas de una banda en el tiempo.

    RADF  -> variable Rad renombrada Cxx.
    CMIPF -> variable CMI renombrada Cxx.
    DQF se conserva cuando está disponible.
    """
    var_fuente = variable_principal(producto)
    escenas = []
    referencia_attrs = None
    referencia_proj = None

    for archivo in sorted(archivos):
        dt = extraer_datetime(archivo)
        if dt is None:
            continue

        with xr.open_dataset(archivo) as ds:
            if var_fuente not in ds:
                print(f"Aviso: {Path(archivo).name} no contiene {var_fuente}")
                continue

            data_vars = {}

            principal = recortar_goes_da(ds[var_fuente], ds, bbox).load()
            if principal.size == 0:
                continue
            data_vars[banda] = principal.astype(np.float32)

            if "DQF" in ds and {"x", "y"}.issubset(ds["DQF"].dims):
                data_vars["DQF"] = recortar_goes_da(ds["DQF"], ds, bbox).load()

            escena = xr.Dataset(data_vars=data_vars)
            escena = escena.expand_dims(time=[np.datetime64(dt)])
            escenas.append(escena)

            if referencia_attrs is None:
                referencia_attrs = dict(ds.attrs)
                referencia_proj = dict(ds["goes_imager_projection"].attrs)

    if not escenas:
        return None

    combinado = xr.concat(
        escenas,
        dim="time",
        data_vars="all",
        coords="minimal",
        compat="override",
        join="outer",
    ).sortby("time")

    combinado["goes_imager_projection"] = xr.DataArray(
        np.int32(0),
        attrs=referencia_proj or {},
    )

    combinado[banda].attrs["grid_mapping"] = "goes_imager_projection"
    combinado[banda].attrs["source_variable"] = var_fuente
    combinado[banda].attrs["source_product"] = producto
    combinado.attrs.update(referencia_attrs or {})
    combinado.attrs.update({
        "title": f"{producto} {banda} recortado y combinado por fecha",
        "product": producto,
        "band": banda,
    })

    return combinado


def guardar_banda_diaria(volcan, fecha, producto, banda, ds_out):
    out_dir = INPUT_BASE / volcan / fecha / producto / banda
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{producto}_{banda}_{fecha}.nc"

    encoding = {
        banda: {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "dtype": "float32",
            "_FillValue": np.float32(-9999.0),
        }
    }

    if "DQF" in ds_out:
        encoding["DQF"] = {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
        }

    ds_out.to_netcdf(out_path, engine="netcdf4", encoding=encoding)
    print(f"Guardado: {out_path}")
    return out_path


# ==========================================================
# PRODUCTO SIN CANAL: ACTPF
# ==========================================================

def _vars_espaciales_actpf(ds):
    """
    Conserva todas las variables 2-D/espaciales dependientes de x,y del ACTPF.
    Así no se codifica a mano un único nombre de variable del producto.
    """
    salida = []
    for nombre, da in ds.data_vars.items():
        if nombre == "goes_imager_projection":
            continue
        if {"x", "y"}.issubset(set(da.dims)):
            salida.append(nombre)
    return salida


def construir_dataset_actpf(archivos, bbox):
    escenas = []
    referencia_attrs = None
    referencia_proj = None
    variables_encontradas = set()

    for archivo in sorted(archivos):
        dt = extraer_datetime(archivo)
        if dt is None:
            continue

        with xr.open_dataset(archivo) as ds:
            rec = recortar_dataset_goes(ds, bbox)
            nombres = _vars_espaciales_actpf(rec)

            if not nombres:
                print(f"Aviso: ACTPF sin variables espaciales en {Path(archivo).name}")
                continue

            escena_vars = {}
            for nombre in nombres:
                da = rec[nombre].load()
                if np.issubdtype(da.dtype, np.floating):
                    da = da.astype(np.float32)
                escena_vars[nombre] = da
                variables_encontradas.add(nombre)

            escena = xr.Dataset(data_vars=escena_vars)
            escena = escena.expand_dims(time=[np.datetime64(dt)])
            escenas.append(escena)

            if referencia_attrs is None:
                referencia_attrs = dict(ds.attrs)
                referencia_proj = dict(ds["goes_imager_projection"].attrs)

    if not escenas:
        return None

    combinado = xr.concat(
        escenas,
        dim="time",
        data_vars="all",
        coords="minimal",
        compat="override",
        join="outer",
    ).sortby("time")

    combinado["goes_imager_projection"] = xr.DataArray(
        np.int32(0), attrs=referencia_proj or {}
    )
    combinado.attrs.update(referencia_attrs or {})
    combinado.attrs.update({
        "title": "ABI-L2-ACTPF recortado y combinado por fecha",
        "product": "ABI-L2-ACTPF",
        "spatial_variables_preserved": ", ".join(sorted(variables_encontradas)),
    })

    return combinado


def guardar_actpf_diario(volcan, fecha, ds_out):
    producto = "ABI-L2-ACTPF"
    out_dir = INPUT_BASE / volcan / fecha / producto
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{producto}_{fecha}.nc"

    encoding = {}
    for nombre, da in ds_out.data_vars.items():
        if nombre == "goes_imager_projection":
            continue
        enc = {"zlib": True, "complevel": 4, "shuffle": True}
        if np.issubdtype(da.dtype, np.floating):
            enc.update({"dtype": "float32", "_FillValue": np.float32(-9999.0)})
        encoding[nombre] = enc

    ds_out.to_netcdf(out_path, engine="netcdf4", encoding=encoding)
    print(f"Guardado: {out_path}")
    return out_path


# ==========================================================
# DESCARGA / PROCESAMIENTO DE LOS TRES PRODUCTOS
# ==========================================================

def procesar_descarga_evento(s3, volcan, fecha, horas_utc, bbox):
    bucket = obtener_bucket_goes(fecha)

    print("\n====================================")
    print(f"VOLCÁN: {volcan}")
    print(f"FECHA:  {fecha}")
    print(f"BUCKET: {bucket}")
    print("PRODUCTOS:", ", ".join(PRODUCTOS_GOES))
    print("====================================")

    tmp_base = INPUT_BASE / ".tmp_goes" / volcan / fecha
    resultados = defaultdict(dict)

    try:
        for producto in PRODUCTOS_GOES:
            print(f"\n========== {producto} ==========")

            descargados = descargar_archivos_producto(
                s3=s3,
                bucket=bucket,
                producto=producto,
                fecha=fecha,
                horas_utc=horas_utc,
                destino_tmp=tmp_base,
            )

            if producto_tiene_bandas(producto):
                for banda in BANDAS_ABI:
                    archivos = descargados.get(banda, [])
                    if not archivos:
                        print(f"Sin archivos para {producto} {banda}")
                        continue

                    print(
                        f"Combinando {producto} {banda}: "
                        f"{len(archivos)} escenas"
                    )
                    ds_out = construir_dataset_banda(
                        archivos=archivos,
                        producto=producto,
                        banda=banda,
                        bbox=bbox,
                    )
                    if ds_out is None:
                        print(f"No se pudo construir {producto} {banda}")
                        continue

                    try:
                        resultados[producto][banda] = guardar_banda_diaria(
                            volcan=volcan,
                            fecha=fecha,
                            producto=producto,
                            banda=banda,
                            ds_out=ds_out,
                        )
                    finally:
                        ds_out.close()

            else:
                if not descargados:
                    print(f"Sin archivos para {producto}")
                    continue

                print(f"Combinando {producto}: {len(descargados)} escenas")
                ds_out = construir_dataset_actpf(descargados, bbox)
                if ds_out is None:
                    print(f"No se pudo construir {producto}")
                    continue

                try:
                    resultados[producto]["ACTPF"] = guardar_actpf_diario(
                        volcan=volcan,
                        fecha=fecha,
                        ds_out=ds_out,
                    )
                finally:
                    ds_out.close()

    finally:
        # Los originales NOAA son temporales: el producto diario recortado queda en RAW_GOES.
        if tmp_base.exists():
            shutil.rmtree(tmp_base, ignore_errors=True)

    return dict(resultados)


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
    transformer = Transformer.from_crs(crs_goes, crs_geo, always_xy=True)

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
                "source_product": PRODUCTO_RGB,
                "source_band": BANDA_NAV,
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

        nav_ds.to_netcdf(out_nav, engine="netcdf4", encoding=encoding)
        nav_ds.close()

    return out_nav


# ==========================================================
# RGB DE CENIZA
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
    # NOAA/NASA
    rNOAA = c15 - c13
    gNOAA = c14 - c11
    bNOAA = c13
    rgbNOAA = np.dstack([
        normalizar(rNOAA, -6.7, 2.6),
        normalizar(gNOAA, -6.0, 6.3),
        normalizar(bNOAA, 243.6, 302.4),
    ])

    # HOTVOLC
    rHOTVOLC = c13 - c15
    gHOTVOLC = c13 - c11
    bHOTVOLC = c13
    rgbHOTVOLC = np.dstack([
        normalizar(rHOTVOLC),
        normalizar(gHOTVOLC),
        normalizar(bHOTVOLC),
    ])

    # Composición espectral asociada al enfoque CNN consultado.
    rCNN = c15 - c13
    gCNN = c13 - c11
    bCNN = c13
    rgbCNN = np.dstack([
        normalizar(rCNN, -4, 2),
        normalizar(gCNN, -4, 5),
        normalizar(bCNN, 243, 303),
    ])

    # Microfísica/Pavolonis: conserva C07 para la diferencia C13-C07.
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


def ruta_banda_diaria(volcan, fecha, banda):
    return (
        INPUT_BASE
        / volcan
        / fecha
        / PRODUCTO_RGB
        / banda
        / f"{PRODUCTO_RGB}_{banda}_{fecha}.nc"
    )


def cargar_bandas_diarias(volcan, fecha):
    datasets = {}

    for banda in BANDAS_RGB:
        ruta = ruta_banda_diaria(volcan, fecha, banda)
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
        print(
            f"Faltan bandas {PRODUCTO_RGB} necesarias para RGB: "
            f"{volcan} {fecha}"
        )
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
                valores[banda] = np.asarray(
                    da.isel(time=idx).values, dtype=float
                )

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

            archivo_nav = (
                resultados
                .get(PRODUCTO_RGB, {})
                .get(BANDA_NAV)
            )
            if GENERAR_NAV_NC and archivo_nav:
                generar_nav_nc(
                    volcan=nombre_volcan,
                    fecha=fecha_evento,
                    archivo_referencia=archivo_nav,
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
        description=(
            "Pipeline GOES: descarga RADF/ACTPF/CMIPF, conserva C01-C16 "
            "donde aplica, combina por fecha y genera RGB de ceniza."
        )
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
