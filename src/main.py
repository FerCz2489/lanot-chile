import argparse


def correr_goes(
    modo,
    volcan=None,
    fecha=None,
    year=None,
    solo_con_hora=False,
    solo_sin_hora=False,
    max_eventos=None
):
    import ash
    ash.run(
        modo=modo,
        volcan=volcan,
        fecha=fecha,
        year=year,
        solo_con_hora=solo_con_hora,
        solo_sin_hora=solo_sin_hora,
        max_eventos=max_eventos
    )


def correr_so2(volcan=None, fecha=None):
    import sentinel5p
    sentinel5p.main(
        fecha=fecha,
        volcan=volcan
    )


def main():
    parser = argparse.ArgumentParser(
        description="LANOT Chile: ejecución principal GOES ceniza y Sentinel-5P SO2."
    )

    parser.add_argument(
        "--producto",
        choices=["goes", "so2", "todo"],
        default="goes",
        help="goes=GOES ceniza, so2=Sentinel-5P SO2, todo=GOES + SO2"
    )

    parser.add_argument(
        "--modo",
        choices=["todo", "descarga", "rgb"],
        default="todo",
        help="Solo aplica para GOES: todo=descarga+rgb, descarga=solo descarga, rgb=solo RGB"
    )

    parser.add_argument(
        "--volcan",
        default=None,
        help=(
            "Volcan a procesar. GOES usa eventos.csv; "
            "SO2 usa volcanes.csv. Si se omite en SO2, usa el shapefile por defecto."
        )
    )

    parser.add_argument(
        "--fecha",
        default=None,
        help="Fecha exacta YYYY-MM-DD. Aplica para GOES y Sentinel-5P SO2."
    )

    parser.add_argument(
        "--year",
        default=None,
        help="GOES: procesa solo eventos de un anio, por ejemplo 2020"
    )

    parser.add_argument(
        "--solo-con-hora",
        action="store_true",
        help="GOES: procesa solo eventos con hora_inicio u hora_fin"
    )

    parser.add_argument(
        "--solo-sin-hora",
        action="store_true",
        help="GOES: procesa solo eventos sin horas definidas, o sea dias completos"
    )

    parser.add_argument(
        "--max-eventos",
        type=int,
        default=None,
        help="GOES: limita la cantidad de filas de eventos.csv a procesar"
    )

    args = parser.parse_args()

    if args.producto == "goes":
        correr_goes(
            args.modo,
            volcan=args.volcan,
            fecha=args.fecha,
            year=args.year,
            solo_con_hora=args.solo_con_hora,
            solo_sin_hora=args.solo_sin_hora,
            max_eventos=args.max_eventos
        )

    elif args.producto == "so2":
        correr_so2(
            volcan=args.volcan,
            fecha=args.fecha
        )

    elif args.producto == "todo":
        correr_goes(
            args.modo,
            volcan=args.volcan,
            fecha=args.fecha,
            year=args.year,
            solo_con_hora=args.solo_con_hora,
            solo_sin_hora=args.solo_sin_hora,
            max_eventos=args.max_eventos
        )

        correr_so2(
            volcan=args.volcan,
            fecha=args.fecha
        )


if __name__ == "__main__":
    main()
