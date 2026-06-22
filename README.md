# LANOT-CHILE

Herramienta para descarga y procesamiento de productos satelitales asociados a emisiones volcánicas en Chile.

Actualmente soporta:

* GOES-16 Full Disk

  * ABI-L1b-RadF
  * ABI-L2-CMIPF
  * ABI-L2-ACTPF
* Sentinel-5P SO₂

Los productos GOES permiten generar distintos RGB de ceniza volcánica, mientras que Sentinel-5P permite visualizar concentraciones de dióxido de azufre (SO₂).

---

# Estructura del proyecto

```text
lanot-chile/
│
├── src/
│   ├── main.py
│   ├── ash.py
│   ├── sentinel5p.py
│   └── config.py
│
├── data/
│   ├── eventos/
│   │   ├── volcanes.csv
│   │   └── eventos.csv
│   │
│   ├── raw/
│   └── outputs/
│
├── README.md
├── requirements.txt
└── lanot-chile.def
```

---

# Clonar el repositorio

Si Git no está instalado:

```bash
sudo apt update
sudo apt install git
```

Clonar el repositorio:

```bash
git clone https://github.com/FerCz2489/lanot-chile.git
```

Entrar al proyecto:

```bash
cd lanot-chile
```

---

# Construcción del contenedor

Instalar Apptainer según la documentación oficial.

Construir la imagen:

```bash
sudo apptainer build lanot-chile.sif lanot-chile.def
```

En caso de tener problemas con fakeroot:

```bash
apptainer build --fakeroot --no-subuid lanot-chile.sif lanot-chile.def
```

Este paso solo se realiza una vez.

---

# Eventos volcánicos

Los eventos se almacenan en:

```text
data/eventos/eventos.csv
```

Formato:

```csv
volcan,fecha,hora_inicio,hora_fin
Villarrica,2023-09-23,08:41,13:24
Peteroa,2018-09-21,,
Chillan,2018-02-02,12:05,12:05
```

Información de volcanes:

```text
data/eventos/volcanes.csv
```

Formato:

```csv
volcan,lat,lon,lon_min,lat_min,lon_max,lat_max
Villarrica,-39.42,-71.93,-74.0,-41.0,-70.0,-38.0
```

---

# GOES-16

## Descarga y procesamiento completo

```bash
apptainer exec lanot-chile.sif python src/main.py \
    --producto goes \
    --modo todo
```

## Solo descarga

```bash
apptainer exec lanot-chile.sif python src/main.py \
    --producto goes \
    --modo descarga
```

## Solo generación de RGB

```bash
apptainer exec lanot-chile.sif python src/main.py \
    --producto goes \
    --modo rgb
```

---

# Filtros para GOES

Para evitar descargar todos los eventos de una sola vez, se pueden usar filtros.

## Filtrar por volcán

```bash
apptainer exec lanot-chile.sif python src/main.py \
    --producto goes \
    --modo descarga \
    --volcan Chillan
```

---

## Filtrar por año

```bash
apptainer exec lanot-chile.sif python src/main.py \
    --producto goes \
    --modo descarga \
    --volcan Chillan \
    --year 2020
```

---

## Filtrar por fecha específica

```bash
apptainer exec lanot-chile.sif python src/main.py \
    --producto goes \
    --modo descarga \
    --volcan Chillan \
    --fecha 2020-04-08
```

---

## Solo eventos con hora definida

Esto evita descargar días completos.

```bash
apptainer exec lanot-chile.sif python src/main.py \
    --producto goes \
    --modo descarga \
    --volcan Chillan \
    --year 2020 \
    --solo-con-hora
```

---

## Solo eventos sin hora definida

Descarga el día completo (00–23 UTC).

```bash
apptainer exec lanot-chile.sif python src/main.py \
    --producto goes \
    --modo descarga \
    --volcan Chillan \
    --year 2020 \
    --solo-sin-hora
```

---

## Limitar número de eventos

Procesa solo las primeras N filas después de aplicar filtros.

```bash
apptainer exec lanot-chile.sif python src/main.py \
    --producto goes \
    --modo descarga \
    --volcan Chillan \
    --year 2020 \
    --max-eventos 3
```

---

## Prueba pequeña recomendada

Ejemplo con un solo evento:

```bash
apptainer exec lanot-chile.sif python src/main.py \
    --producto goes \
    --modo descarga \
    --volcan Chillan \
    --fecha 2020-04-08 \
    --solo-con-hora \
    --max-eventos 1
```

Después generar RGB:

```bash
apptainer exec lanot-chile.sif python src/main.py \
    --producto goes \
    --modo rgb \
    --volcan Chillan \
    --fecha 2020-04-08 \
    --solo-con-hora \
    --max-eventos 1
```

---

# Productos descargados

GOES:

* ABI-L1b-RadF
* ABI-L2-CMIPF
* ABI-L2-ACTPF

RGB generados:

* NOAA/NASA Ash RGB
* HOTVOLC
* CNN Ash RGB
* MICROPHYSICS_PAVOLONIS

---

# Sentinel-5P SO₂

Definir credenciales de Copernicus:

```bash
export COPERNICUS_USER="usuario"
export COPERNICUS_PASSWORD="password"
```

Ejecutar:

```bash
apptainer exec lanot-chile.sif python src/main.py \
    --producto so2
```

---

# Salidas

GOES:

```text
data/raw/GOES/
data/outputs/rgb_ash/
```

Sentinel-5P:

```text
data/raw/S5P/
data/outputs/so2/
```

---

# Notas

* Los datos crudos no se almacenan en GitHub.
* Los productos generados no se almacenan en GitHub.
* El archivo `.sif` no se almacena en GitHub.
* Los eventos volcánicos pueden añadirse modificando únicamente los archivos CSV.
* No es necesario modificar el código para agregar nuevos volcanes o nuevas fechas; basta con actualizar `volcanes.csv` y `eventos.csv`.
* Para pruebas se recomienda usar `--volcan`, `--year`, `--fecha`, `--solo-con-hora` y `--max-eventos`.
