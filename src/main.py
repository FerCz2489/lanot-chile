import argparse


def correr_goes(modo):
    import ash
    ash.run(modo=modo)


def correr_so2():
    import sentinel5p
    sentinel5p.main()


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

    args = parser.parse_args()

    if args.producto == "goes":
        correr_goes(args.modo)

    elif args.producto == "so2":
        correr_so2()

    elif args.producto == "todo":
        correr_goes(args.modo)
        correr_so2()


if __name__ == "__main__":
    main()
