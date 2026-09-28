from pathlib import Path



from datetime import datetime, timedelta, timezone



from urllib.parse import quote



from mpl_toolkits.axes_grid1.inset_locator import inset_axes



from matplotlib.patches import Rectangle



import os



import csv



import subprocess







import requests



import numpy as np



import xarray as xr



import matplotlib.pyplot as plt



import geopandas as gpd



import rasterio



from rasterio.transform import from_origin







from scipy.interpolate import griddata







from config import RAW_S5P, OUT_SO2











# ============================================================



# SENTINEL-5P SO2:



# Copernicus L2 -> HARP bin_spatial -> grilla regular -> PNG



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



NAME_NRTI = "S5P_NRTI_L2__SO2"



NAME_OFFL = "S5P_OFFL_L2__SO2"







SRC_DIR = Path(__file__).resolve().parent



BASE_DIR = SRC_DIR.parent



SHAPE_ESTUDIO = SRC_DIR / "campo villarrica.shp"



SHAPE_CHILE = SRC_DIR / "REGIONES" / "REGIONES_v1.shp"
SHAPE_COMUNAS = SRC_DIR / "COMUNAS" / "COMUNAS_v1.shp"



def agregar_minimapa(ax, bounds, lon_volcan, lat_volcan):
    """Ubica claramente el dominio dentro de Chile."""
    if not SHAPE_CHILE.exists():
        print("ADVERTENCIA: no existe la capa REGIONES; se omite el minimapa.")
        return

    chile = gpd.read_file(SHAPE_CHILE)
    chile = chile.set_crs("EPSG:4326") if chile.crs is None else chile.to_crs("EPSG:4326")

    ax_inset = inset_axes(
        ax, width="14%", height="44%",
        loc="lower left", borderpad=1.5
    )
    ax_inset.set_facecolor("white")

    chile.plot(
        ax=ax_inset, facecolor="0.88", edgecolor="0.25",
        linewidth=0.5, zorder=1
    )

    minx, miny, maxx, maxy = bounds
    ax_inset.add_patch(Rectangle(
        (minx, miny), maxx-minx, maxy-miny,
        facecolor="red", edgecolor="red",
        alpha=0.35, linewidth=2.2, zorder=10
    ))
    ax_inset.scatter(
        lon_volcan, lat_volcan, marker="^", s=45,
        color="deepskyblue", edgecolor="black",
        linewidth=0.7, zorder=20
    )

    xmin, ymin, xmax, ymax = chile.total_bounds
    ax_inset.set_xlim(xmin - 0.5, xmax + 0.5)
    ax_inset.set_ylim(ymin - 0.5, ymax + 0.5)
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax_inset.set_xlabel("")
    ax_inset.set_ylabel("")
    ax_inset.set_title("Ubicación en Chile", fontsize=8, fontweight="bold", pad=3)

    for spine in ax_inset.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor("black")


def agregar_divisiones_administrativas(ax, bounds):
    """Superpone comunas punteadas y regiones sobre el raster."""
    minx, miny, maxx, maxy = bounds
    bbox = (minx, miny, maxx, maxy)

    if SHAPE_COMUNAS.exists():
        comunas = gpd.read_file(SHAPE_COMUNAS, bbox=bbox)
        if not comunas.empty:
            comunas = comunas.set_crs("EPSG:4326") if comunas.crs is None else comunas.to_crs("EPSG:4326")
            comunas.boundary.plot(
                ax=ax, color="white", linewidth=0.6,
                linestyle=(0, (2, 3)), alpha=0.75, zorder=24
            )

    if SHAPE_CHILE.exists():
        regiones = gpd.read_file(SHAPE_CHILE, bbox=bbox)
        if not regiones.empty:
            regiones = regiones.set_crs("EPSG:4326") if regiones.crs is None else regiones.to_crs("EPSG:4326")
            regiones.boundary.plot(
                ax=ax, color="white", linewidth=1.25,
                alpha=0.95, zorder=25
            )


VOLCANES_CSV = BASE_DIR / "data" / "eventos" / "volcanes.csv"







VILLARRICA_LAT = -39.420



VILLARRICA_LON = -71.930







# Shape completo + alrededores



MARGEN_FRACCION = 0.08







# Resolucion HARP en grados (~0.05 = ~5 km)



RESOLUCION_GRADOS = 0.05







# QA HARP: qa_value 0..1 pasa a validity 0..100



QA_MIN_HARP = 50







# HARP SO2_type:



# 0=no_detection, 1=so2_detected, 2=volcanic_detection,



# 3=detection_near_anthropogenic_source, 4=detection_at_high_sza



# El tipo 4 puede contener falsos positivos por SZA alto.



EXCLUIR_SO2_TYPE_4 = True







# Para --latest



SEARCH_DAYS_BACK = 7







# Para fecha/hora especifica



SEARCH_HOURS_AROUND = 3







# Visual



PERCENTIL_VMAX = 99.5







# Si quedan celdas sin observacion tras bin_spatial,



# rellenarlas visualmente para evitar blanco.



RELLENAR_HUECOS_VISUALES = True











# ============================================================



# SHAPE Y REGION



# ============================================================







def cargar_zona_estudio():



    if not SHAPE_ESTUDIO.exists():



        raise FileNotFoundError(f"No existe: {SHAPE_ESTUDIO}")







    gdf = gpd.read_file(SHAPE_ESTUDIO)







    if gdf.empty:



        raise RuntimeError("El shapefile esta vacio.")







    if gdf.crs is None:



        print("ADVERTENCIA: shape sin CRS; se asume EPSG:4326.")



        gdf = gdf.set_crs("EPSG:4326")



    else:



        gdf = gdf.to_crs("EPSG:4326")







    try:



        geom = gdf.geometry.union_all()



    except AttributeError:



        geom = gdf.geometry.unary_union







    minx, miny, maxx, maxy = geom.bounds







    mx = (maxx - minx) * MARGEN_FRACCION



    my = (maxy - miny) * MARGEN_FRACCION







    bounds_mapa = (



        max(-180.0, minx - mx),



        max(-90.0, miny - my),



        min(180.0, maxx + mx),



        min(90.0, maxy + my),



    )







    print("\n====================================")



    print("ZONA DE ESTUDIO")



    print("====================================")



    print("Shape:", SHAPE_ESTUDIO)



    print("Bounds shape:", geom.bounds)



    print("Bounds mapa:", bounds_mapa)







    return geom, bounds_mapa











def cargar_volcan_csv(nombre_volcan):



    """



    Añadido opcional: busca un volcan en data/eventos/volcanes.csv.



    Si no se usa --volcan, el flujo normal sigue usando el shapefile.



    """



    if not VOLCANES_CSV.exists():



        raise FileNotFoundError(f"No existe: {VOLCANES_CSV}")







    buscado = nombre_volcan.strip().casefold()







    with open(VOLCANES_CSV, newline="", encoding="utf-8") as f:



        reader = csv.DictReader(f)



        for row in reader:



            if row["volcan"].strip().casefold() == buscado:



                nombre = row["volcan"].strip()



                lat = float(row["lat"])



                lon = float(row["lon"])



                bounds = (



                    float(row["lon_min"]),



                    float(row["lat_min"]),



                    float(row["lon_max"]),



                    float(row["lat_max"]),



                )







                print("\n====================================")



                print("ZONA DE ESTUDIO DESDE volcanes.csv")



                print("====================================")



                print("Volcan:", nombre)



                print("Centro:", lon, lat)



                print("Bounds mapa:", bounds)







                return nombre, lat, lon, bounds







    raise ValueError(



        f"No encontre el volcan '{nombre_volcan}' en {VOLCANES_CSV}"



    )











def bbox_wkt(bounds):



    minx, miny, maxx, maxy = bounds



    return (



        f"POLYGON(("



        f"{minx} {miny},"



        f"{maxx} {miny},"



        f"{maxx} {maxy},"



        f"{minx} {maxy},"



        f"{minx} {miny}"



        f"))"



    )











# ============================================================



# COPERNICUS



# ============================================================







def obtener_token():



    if not COPERNICUS_USER or not COPERNICUS_PASSWORD:



        raise RuntimeError(



            "Faltan COPERNICUS_USER y COPERNICUS_PASSWORD."



        )







    r = requests.post(



        TOKEN_URL,



        data={



            "client_id": "cdse-public",



            "username": COPERNICUS_USER,



            "password": COPERNICUS_PASSWORD,



            "grant_type": "password",



        },



        timeout=60,



    )



    r.raise_for_status()



    return r.json()["access_token"]











def buscar_productos(inicio, fin, bounds, top=100, name_contains=NAME_NRTI):



    ini = inicio.strftime("%Y-%m-%dT%H:%M:%S.000Z")



    end = fin.strftime("%Y-%m-%dT%H:%M:%S.000Z")



    wkt = bbox_wkt(bounds)







    filtro = (



        f"Collection/Name eq '{COLLECTION}' "



        f"and contains(Name,'{name_contains}') "



        f"and ContentDate/Start ge {ini} "



        f"and ContentDate/Start le {end} "



        f"and OData.CSC.Intersects("



        f"area=geography'SRID=4326;{wkt}')"



    )







    url = (



        CATALOG_URL



        + f"?$filter={quote(filtro, safe='()/,$=; ')}"



        + "&$orderby=ContentDate/Start desc"



        + f"&$top={top}"



    )







    r = requests.get(url, timeout=90)



    r.raise_for_status()



    return r.json().get("value", [])











def fecha_producto(producto):



    return datetime.fromisoformat(



        producto["ContentDate"]["Start"].replace("Z", "+00:00")



    )







def obtener_intervalo_productos(productos):



    """



    Obtiene el intervalo UTC de los productos Sentinel-5P utilizados.







    Si se usa un solo swath, devuelve una sola hora.



    Si se usan varios, devuelve desde la primera hasta la ultima hora.



    """



    fechas = sorted(fecha_producto(p) for p in productos)







    inicio = fechas[0]



    fin = fechas[-1]







    if len(fechas) == 1:



        texto = inicio.strftime("%H:%M UTC")



    else:



        texto = (



            f"{inicio.strftime('%H:%M')}–"



            f"{fin.strftime('%H:%M UTC')}"



        )







    print("\n====================================")



    print("PASO SENTINEL-5P")



    print("====================================")



    print("Productos utilizados:", len(productos))



    print("Inicio:", inicio.strftime("%Y-%m-%d %H:%M:%S UTC"))



    print("Fin:", fin.strftime("%Y-%m-%d %H:%M:%S UTC"))



    print("Texto para figura:", texto)







    return inicio, fin, texto







def buscar_productos_fecha(inicio, fin, bounds, top=100):



    """Para fechas explicitas prioriza OFFL; NRTI queda como respaldo."""



    print("\nBuscando Sentinel-5P SO2 OFFL...")



    productos = buscar_productos(inicio, fin, bounds, top=top, name_contains=NAME_OFFL)



    if productos:



        print("Producto usado: OFFL")



        return productos







    print("No encontre OFFL. Intentando NRTI...")



    productos = buscar_productos(inicio, fin, bounds, top=top, name_contains=NAME_NRTI)



    if productos:



        print("Producto usado: NRTI")



    return productos











def seleccionar_productos(fecha, hora, bounds):



    """



    Sin fecha:



      toma todos los swaths de la fecha del producto mas reciente



      que intersecta la region.







    Con fecha:



      si hora=None, toma todos los swaths de ese dia.



      si hora se indica, toma el swath temporalmente mas cercano.



    """



    if fecha is None:



        ahora = datetime.now(timezone.utc)



        inicio = ahora - timedelta(days=SEARCH_DAYS_BACK)







        encontrados = buscar_productos(



            inicio, ahora, bounds, top=100, name_contains=NAME_NRTI



        )







        if not encontrados:



            raise RuntimeError("No encontre L2 SO2 reciente.")







        dia = fecha_producto(encontrados[0]).date()







        inicio_dia = datetime.combine(



            dia, datetime.min.time(), tzinfo=timezone.utc



        )



        fin_dia = inicio_dia + timedelta(days=1)







        productos = buscar_productos(



            inicio_dia, fin_dia, bounds, top=100, name_contains=NAME_NRTI



        )







        print("\nFecha L2 mas reciente:", dia)



        print("Swaths encontrados:", len(productos))



        return productos







    dia = datetime.strptime(fecha, "%Y-%m-%d").date()







    if hora is None:



        inicio = datetime.combine(



            dia, datetime.min.time(), tzinfo=timezone.utc



        )



        fin = inicio + timedelta(days=1)







        productos = buscar_productos_fecha(



            inicio, fin, bounds, top=100



        )







        if not productos:



            raise RuntimeError(f"No hay L2 SO2 OFFL ni NRTI para {fecha} en esta region.")







        print("\nFecha solicitada:", fecha)



        print("Swaths encontrados:", len(productos))



        return productos







    objetivo = datetime.strptime(



        f"{fecha} {hora:02d}:00:00",



        "%Y-%m-%d %H:%M:%S",



    ).replace(tzinfo=timezone.utc)







    productos = buscar_productos_fecha(



        objetivo - timedelta(hours=SEARCH_HOURS_AROUND),



        objetivo + timedelta(hours=SEARCH_HOURS_AROUND),



        bounds,



        top=100,



    )







    if not productos:



        raise RuntimeError(



            f"No hay L2 SO2 OFFL ni NRTI cerca de {fecha} {hora:02d}:00 UTC."



        )







    elegido = min(



        productos,



        key=lambda p: abs(



            (fecha_producto(p) - objetivo).total_seconds()



        ),



    )







    print("\nProducto mas cercano:", elegido["Name"])



    print("Fecha:", elegido["ContentDate"]["Start"])







    return [elegido]











def descargar_producto(producto, token):



    RAW_S5P.mkdir(parents=True, exist_ok=True)







    nombre = producto["Name"]



    if not nombre.endswith(".nc"):



        nombre += ".nc"







    salida = RAW_S5P / nombre







    if salida.exists() and salida.stat().st_size > 0:



        print("Ya existe:", salida.name)



        return salida







    url = f"{DOWNLOAD_URL}({producto['Id']})/$value"



    tmp = salida.with_suffix(".nc.part")







    print("Descargando:", salida.name)







    with requests.get(



        url,



        headers={"Authorization": f"Bearer {token}"},



        stream=True,



        timeout=300,



    ) as r:



        r.raise_for_status()







        with open(tmp, "wb") as f:



            for chunk in r.iter_content(1024 * 1024):



                if chunk:



                    f.write(chunk)







    tmp.replace(salida)



    return salida











# ============================================================



# HARP: L2 -> GRILLA



# ============================================================







def construir_edges(vmin, vmax, paso):



    inicio = np.floor(vmin / paso) * paso



    fin = np.ceil(vmax / paso) * paso







    edges = np.arange(



        inicio,



        fin + paso * 0.5,



        paso,



        dtype=float,



    )







    return edges











def lista_harp(valores):



    return "(" + ",".join(f"{v:.6f}" for v in valores) + ")"











def convertir_con_harp(archivos_l2, bounds, fecha_tag):



    """



    1) Convierte cada L2 a HARP y filtra QA/region.



    2) Combina los productos.



    3) bin_spatial usando latitude_bounds/longitude_bounds.



    """



    minx, miny, maxx, maxy = bounds







    lat_edges = construir_edges(



        miny, maxy, RESOLUCION_GRADOS



    )



    lon_edges = construir_edges(



        minx, maxx, RESOLUCION_GRADOS



    )







    harp_dir = RAW_S5P / "harp"



    harp_dir.mkdir(parents=True, exist_ok=True)







    temporales = []







    # Filtramos por centros de pixel un poco mas amplio que el mapa.



    # Filtro fundamentado en el PUM + ingestion HARP:



    # - QA >= 0.5  -> validity >= 50



    # - excluir SO2_type == 4 (deteccion a SZA alto / potencial falso positivo)



    filtro_tipo = "SO2_type!=4;" if EXCLUIR_SO2_TYPE_4 else ""







    operaciones_pre = (



        f"SO2_column_number_density_validity>={QA_MIN_HARP};"



        f"{filtro_tipo}"



        f"latitude>={lat_edges[0]};"



        f"latitude<={lat_edges[-1]};"



        f"longitude>={lon_edges[0]};"



        f"longitude<={lon_edges[-1]};"



        "keep(datetime_start,latitude,longitude,"



        "latitude_bounds,longitude_bounds,"



        "SO2_column_number_density,"



        "SO2_column_number_density_validity)"



    )







    for i, archivo in enumerate(archivos_l2, start=1):



        tmp = harp_dir / f"tmp_{fecha_tag}_{i:02d}.nc"







        cmd = [



            "harpconvert",



            "-a", operaciones_pre,



            str(archivo),



            str(tmp),



        ]







        print(f"HARP ingest {i}/{len(archivos_l2)}...")







        # Algunos swaths intersectan el bbox del catalogo, pero despues



        # del filtro QA + recorte HARP pueden quedar completamente vacios.



        # Eso NO debe detener el mosaico: simplemente se omiten.



        resultado = subprocess.run(



            cmd,



            check=False,



            capture_output=True,



            text=True,



        )







        if resultado.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:



            mensaje = (resultado.stderr or resultado.stdout or "").strip()







            if "product is empty" in mensaje.lower():



                print(f"  -> VACIO despues de QA/recorte. Se omite.")



            else:



                print(f"  -> HARP no pudo usar este swath. Se omite.")



                if mensaje:



                    print("     ", mensaje.splitlines()[-1])







            try:



                tmp.unlink()



            except OSError:



                pass







            continue







        print("  -> OK")



        temporales.append(tmp)







    if not temporales:



        raise RuntimeError("Todos los swaths quedaron vacios despues del filtro QA/recorte HARP.")







    combinado = harp_dir / f"SO2_{fecha_tag}_L2_COMBINADO.nc"







    if len(temporales) == 1:



        # harpconvert para asegurar un archivo separado.



        subprocess.run(



            [



                "harpconvert",



                str(temporales[0]),



                str(combinado),



            ],



            check=True,



        )



    else:



        # harpmerge concatena los productos en la dimension temporal.



        subprocess.run(



            [



                "harpmerge",



                *[str(p) for p in temporales],



                str(combinado),



            ],



            check=True,



        )







    salida_l3 = harp_dir / f"SO2_{fecha_tag}_HARP_L3.nc"







    # bin_spatial ya genera el producto espacial gridded.



    # NO aplicar keep(latitude,longitude,...) despues: en esta etapa



    # HARP puede representar la grilla mediante dimensiones/variables



    # espaciales distintas y ese keep provoca:



    # "cannot keep non-existent variable latitude".



    operaciones_bin = (



        f"bin_spatial("



        f"{lista_harp(lat_edges)},"



        f"{lista_harp(lon_edges)}"



        f")"



    )







    print("\nFiltros HARP aplicados:")



    print(f"  validity >= {QA_MIN_HARP}")



    print(f"  excluir SO2_type == 4: {EXCLUIR_SO2_TYPE_4}")



    print("\nHARP bin_spatial...")



    subprocess.run(



        [



            "harpconvert",



            "-a", operaciones_bin,



            str(combinado),



            str(salida_l3),



        ],



        check=True,



    )







    # Limpieza de temporales.



    for p in temporales:



        try:



            p.unlink()



        except OSError:



            pass







    try:



        combinado.unlink()



    except OSError:



        pass







    print("L3 HARP:", salida_l3)



    return salida_l3











# ============================================================



# LEER L3 HARP



# ============================================================







def leer_harp_l3(ruta):



    ds = xr.open_dataset(ruta)







    try:



        # HARP bin_spatial puede no guardar latitude/longitude como



        # variables; en ese caso guarda solamente los bordes de cada celda.



        if "latitude" in ds.variables:



            lat = np.asarray(ds["latitude"].values).squeeze()



        elif "latitude_bounds" in ds.variables:



            lat_bounds = np.asarray(ds["latitude_bounds"].values, dtype=float)



            lat = np.nanmean(lat_bounds, axis=1)



        else:



            raise RuntimeError(



                "El L3 HARP no contiene latitude ni latitude_bounds."



            )







        if "longitude" in ds.variables:



            lon = np.asarray(ds["longitude"].values).squeeze()



        elif "longitude_bounds" in ds.variables:



            lon_bounds = np.asarray(ds["longitude_bounds"].values, dtype=float)



            lon = np.nanmean(lon_bounds, axis=1)



        else:



            raise RuntimeError(



                "El L3 HARP no contiene longitude ni longitude_bounds."



            )







        da = ds["SO2_column_number_density"]







        # Salida observada de HARP:



        # (time=1, latitude, longitude)



        if "time" in da.dims:



            da = da.isel(time=0)







        da = da.squeeze(drop=True)



        z = np.asarray(da.values, dtype=float)







        if z.shape == (lon.size, lat.size):



            z = z.T







        if z.shape != (lat.size, lon.size):



            raise RuntimeError(



                f"Shape inesperado HARP: z={z.shape}, "



                f"lat={lat.size}, lon={lon.size}"



            )







        units = da.attrs.get("units", "mol/m^2")







        print("\n========== L3 HARP LEIDO ==========")



        print("SO2:", z.shape)



        print("Lat:", lat.size, float(np.nanmin(lat)), "a", float(np.nanmax(lat)))



        print("Lon:", lon.size, float(np.nanmin(lon)), "a", float(np.nanmax(lon)))



        print("Valores SO2 validos:", int(np.isfinite(z).sum()), "/", int(z.size))



        print("Unidades:", units)







    finally:



        ds.close()







    return lon, lat, z, units











# ============================================================



# RELLENO VISUAL



# ============================================================







def rellenar_huecos(lon, lat, z):



    """



    Rellena SOLO huecos pequenos/locales del L3 HARP.







    No interpola entre orbitas separadas ni inventa valores a grandes



    distancias. Esto evita triangulos/rayas artificiales.



    """



    if not RELLENAR_HUECOS_VISUALES:



        return z







    from scipy.ndimage import distance_transform_edt







    z = np.asarray(z, dtype=float)



    valid = np.isfinite(z)







    print(



        "Celdas HARP validas:",



        int(valid.sum()),



        "/",



        int(z.size),



    )







    if valid.sum() == 0 or valid.all():



        return z







    # Distancia, en numero de celdas, al dato valido mas cercano.



    dist, indices = distance_transform_edt(



        ~valid,



        return_distances=True,



        return_indices=True,



    )







    # Solo rellenamos huecos muy pequenos: maximo 2 celdas (~0.1 grados



    # con la resolucion actual de 0.05 grados).



    MAX_DIST_CELDAS = 2.0



    rellenables = (~valid) & (dist <= MAX_DIST_CELDAS)







    salida = z.copy()



    nearest = z[tuple(indices)]



    salida[rellenables] = nearest[rellenables]







    print("Huecos locales rellenados:", int(rellenables.sum()))



    print("Huecos grandes conservados sin inventar:", int((~np.isfinite(salida)).sum()))







    return salida











# ============================================================



# GEOTIFF



# ============================================================







def guardar_geotiff(lon, lat, z, fecha_tag):



    """Guarda la malla HARP como GeoTIFF EPSG:4326."""



    lon = np.asarray(lon, dtype=float)



    lat = np.asarray(lat, dtype=float)



    z = np.asarray(z, dtype=float)







    dx = abs(float(np.nanmedian(np.diff(lon))))



    dy = abs(float(np.nanmedian(np.diff(lat))))







    if lat[0] < lat[-1]:



        z_tif = np.flipud(z)



        north = float(lat[-1] + dy / 2.0)



    else:



        z_tif = z.copy()



        north = float(lat[0] + dy / 2.0)







    west = float(np.nanmin(lon) - dx / 2.0)



    transform = from_origin(west, north, dx, dy)







    out_dir = OUT_SO2 / fecha_tag



    out_dir.mkdir(parents=True, exist_ok=True)







    salida_tif = out_dir / f"SO2_HARP_L3_ESCALA_FIJA_{fecha_tag}.tif"







    nodata = -9999.0



    data_out = np.where(np.isfinite(z_tif), z_tif, nodata).astype("float32")







    with rasterio.open(



        salida_tif,



        "w",



        driver="GTiff",



        height=data_out.shape[0],



        width=data_out.shape[1],



        count=1,



        dtype="float32",



        crs="EPSG:4326",



        transform=transform,



        nodata=nodata,



        compress="deflate",



        predictor=3,



        tiled=True,



    ) as dst:



        dst.write(data_out, 1)



        dst.set_band_description(1, "SO2 vertical column")



        dst.update_tags(



            1,



            units="mol m-2",



            source="Sentinel-5P TROPOMI L2 -> HARP bin_spatial",



            qa_filter="validity >= 50; SO2_type != 4",



        )







    print("\n====================================")



    print("GEOTIFF GENERADO")



    print("====================================")



    print(salida_tif)







    return salida_tif











# ============================================================



# PLOT



# ============================================================







def dibujar_shape(ax, geom):



    if geom.geom_type == "Polygon":



        x, y = geom.exterior.xy



        ax.plot(x, y, color="black", linewidth=1.4, zorder=20)







    elif geom.geom_type == "MultiPolygon":



        for poly in geom.geoms:



            x, y = poly.exterior.xy



            ax.plot(x, y, color="black", linewidth=1.4, zorder=20)







def agregar_barra_escala(ax, bounds, km=50):



    """



    Barra de escala aproximada para un mapa en EPSG:4326.







    Convierte los km a grados de longitud usando la latitud



    central del dominio.



    """



    minx, miny, maxx, maxy = bounds







    lat_centro = (miny + maxy) / 2.0







    # Longitud aproximada de 1 grado de longitud a esta latitud.



    km_por_grado_lon = 111.32 * np.cos(np.deg2rad(lat_centro))



    grados_lon = km / km_por_grado_lon







    ancho = maxx - minx



    alto = maxy - miny






    # Esquina inferior derecha, arriba de la caja de resolución
    x0 = maxx - grados_lon - 0.04 * ancho
    y0 = miny + 0.08 * alto


    # Fondo blanco semitransparente de la escala grafica.
    fondo_escala = Rectangle(
    (x0 - 0.012 * ancho, y0 - 0.020 * alto),
    grados_lon + 0.024 * ancho,
    0.065 * alto,
    facecolor="white",
    edgecolor="black",
    linewidth=0.8,
    alpha=0.80,
    zorder=49,
)
    ax.add_patch(fondo_escala)







    ax.plot(



        [x0, x0 + grados_lon],



        [y0, y0],



        color="black",



        linewidth=4,



        solid_capstyle="butt",



        zorder=50,



    )







    # Marcas de los extremos.



    marca = 0.012 * alto







    ax.plot(



        [x0, x0],



        [y0 - marca, y0 + marca],



        color="black",



        linewidth=2,



        zorder=50,



    )







    ax.plot(



        [x0 + grados_lon, x0 + grados_lon],



        [y0 - marca, y0 + marca],



        color="black",



        linewidth=2,



        zorder=50,



    )







    ax.text(



        x0 + grados_lon / 2,



        y0 + 0.018 * alto,



        f"{km} km",



        ha="center",



        va="bottom",



        fontsize=10,



        fontweight="bold",



        color="black",



        zorder=50,



    )



def agregar_info_resolucion(ax, bounds):

    """

    Muestra el tamaño aproximado de una celda HARP

    para la latitud central del dominio.

    """

    minx, miny, maxx, maxy = bounds

    lat_centro = (miny + maxy) / 2.0



    km_lat = RESOLUCION_GRADOS * 111.32

    km_lon = (

        RESOLUCION_GRADOS

        * 111.32

        * np.cos(np.deg2rad(lat_centro))

    )



    area = km_lat * km_lon



    texto = (

        f"Rejilla L3 HARP: "

        f"{RESOLUCION_GRADOS:.2f}° × {RESOLUCION_GRADOS:.2f}°\n"

        f"≈ {km_lat:.1f} × {km_lon:.1f} km "

        f"(≈ {area:.0f} km² por celda)"

    )



    ax.text(

        0.99,

        0.015,

        texto,

        transform=ax.transAxes,

        ha="right",

        va="bottom",

        fontsize=9,

        bbox=dict(

            boxstyle="round,pad=0.4",

            facecolor="white",

            edgecolor="black",

            alpha=0.85,

        ),

        zorder=60,

    )



def graficar(



    ruta_harp,



    fecha_tag,



    geom,



    bounds,



    nombre_volcan="Villarrica",



    lat_volcan=VILLARRICA_LAT,



    lon_volcan=VILLARRICA_LON,



    fecha_inicio=None,



    fecha_fin=None,



    texto_hora=None,



):



    lon, lat, z, units = leer_harp_l3(ruta_harp)







    z = rellenar_huecos(lon, lat, z)







    guardar_geotiff(



        lon=lon,



        lat=lat,



        z=z,



        fecha_tag=fecha_tag,



    )







    valid = z[np.isfinite(z)]







    if valid.size == 0:



        raise RuntimeError(



            "El bin_spatial no produjo datos SO2 validos en la region."



        )







    # SO2 puede tener ruido negativo; para visualizacion lo dejamos en 0.



    z_plot = np.array(z, copy=True)



    z_plot[z_plot < 0] = 0.0







    positivos = z_plot[



        np.isfinite(z_plot)



    ]







    # PRUEBA 3:



    # NO estirar la escala con percentiles.



    # Usamos una escala fija 0-0.01 mol/m^2, como en las primeras



    # visualizaciones y mucho menos sensible al ruido de fondo.



    vmax = 0.01







    X, Y = np.meshgrid(lon, lat)







    fig, ax = plt.subplots(figsize=(14, 8))







    # Raster L3 real. pcolormesh respeta la malla HARP y evita



    # triangulaciones/contornos artificiales entre celdas.



    z_plot = np.ma.masked_invalid(z_plot)







    cf = ax.pcolormesh(



        X,



        Y,



        z_plot,



        cmap="jet",



        vmin=0.0,



        vmax=vmax,



        shading="nearest",



        rasterized=True,



        alpha=0.82,
        zorder=10,



    )







    cbar = fig.colorbar(cf, ax=ax, pad=0.025)



    cbar.set_label(



        f"SO$_2$ vertical column ({units})"



    )







    if geom is not None:



        dibujar_shape(ax, geom)







    ax.scatter(



        lon_volcan,



        lat_volcan,



        marker="^",



        s=130,



        color="deepskyblue",



        edgecolor="black",



        linewidth=1.0,



        label=nombre_volcan,



        zorder=30,



    )







    minx, miny, maxx, maxy = bounds



    ax.set_xlim(minx, maxx)



    ax.set_ylim(miny, maxy)

    agregar_divisiones_administrativas(
        ax=ax,
        bounds=bounds,
    )




    # Referencias cartograficas



    agregar_barra_escala(



        ax=ax,



        bounds=bounds,



        km=50,



    )







    agregar_info_resolucion(



        ax=ax,



        bounds=bounds,



    )














    ax.set_xlabel("Longitud")



    ax.set_ylabel("Latitud")



    fecha_texto = (



        fecha_inicio.strftime("%d/%m/%Y")



        if fecha_inicio is not None



        else fecha_tag



    )







    ax.set_title(



        f"Sentinel-5P/TROPOMI — Columna vertical de SO$_2$\n"



        f"{nombre_volcan} | {fecha_texto} | {texto_hora}",



        fontsize=14,



        fontweight="bold",



    )



    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
    )



    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.35)







    out_dir = OUT_SO2 / fecha_tag



    out_dir.mkdir(parents=True, exist_ok=True)







    salida = (



        out_dir



        / f"SO2_HARP_L3_ESCALA_FIJA_{fecha_tag}.png"



    )







    plt.tight_layout()



    plt.savefig(



        salida,



        dpi=250,



        bbox_inches="tight",



    )



    plt.close(fig)







    print("\n====================================")



    print("PNG GENERADO")



    print("====================================")



    print(salida)











# ============================================================



# MAIN



# ============================================================







def main(fecha=None, hora=None, volcan=None):



    """



    Compatible con tu main.py actual:



        sentinel5p.main(fecha=args.fecha, hora=args.hora)







    Sin --fecha:



        usa la fecha L2 mas reciente y junta todos los swaths del dia.







    --fecha YYYY-MM-DD:



        junta todos los swaths del dia.







    --fecha YYYY-MM-DD --hora HH:



        usa el swath mas cercano a esa hora.







    Como recordatorio, modificamos el plot a un área más amplia



    """







    if volcan is None:



        # Flujo original/default: usa el shapefile de Villarrica.



        geom, bounds = cargar_zona_estudio()



        nombre_volcan = "Villarrica"



        lat_volcan = VILLARRICA_LAT



        lon_volcan = VILLARRICA_LON



    else:



        # Añadido opcional: --volcan usa lat/lon/bbox de volcanes.csv.



        nombre_volcan, lat_volcan, lon_volcan, bounds = cargar_volcan_csv(volcan)



        geom = None







    productos = seleccionar_productos(



        fecha=fecha,



        hora=hora,



        bounds=bounds,



    )







    if not productos:



        raise RuntimeError("No se encontraron productos L2.")







    fecha_inicio, fecha_fin, texto_hora = obtener_intervalo_productos(



        productos



    )



    token = obtener_token()







    archivos = [



        descargar_producto(p, token)



        for p in productos



    ]







    if fecha is not None:



        fecha_tag = fecha



        if hora is not None:



            fecha_tag += f"_H{hora:02d}"



    else:



        fecha_tag = fecha_producto(productos[0]).strftime(



            "%Y-%m-%d"



        )







    harp_l3 = convertir_con_harp(



        archivos_l2=archivos,



        bounds=bounds,



        fecha_tag=fecha_tag.replace(":", "-"),



    )







    graficar(



        ruta_harp=harp_l3,



        fecha_tag=fecha_tag,



        geom=geom,



        bounds=bounds,



        nombre_volcan=nombre_volcan,



        lat_volcan=lat_volcan,



        lon_volcan=lon_volcan,



        fecha_inicio=fecha_inicio,



        fecha_fin=fecha_fin,



        texto_hora=texto_hora,



    )







    print("\nProceso terminado.")



    print("L2:", RAW_S5P)



    print("L3 HARP:", harp_l3)



    print("PNG:", OUT_SO2)











if __name__ == "__main__":



    main()
