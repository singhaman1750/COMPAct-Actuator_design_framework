import sys

GEARBOX_DISPATCH = {
    "sspg": "Opt_singleStagePlanetaryGBOptimization",
    "dspg": "Opt_doubleStagePlanetaryGBOptimization",
    "cpg": "Opt_compoundPlanetaryGBOptimization",
    "wpg": "Opt_wolfromPlanetaryGBOptimization",
}

def main(motor, gearbox_type, gear_ratio=0):
    if gearbox_type not in GEARBOX_DISPATCH:
        raise ValueError(f"Unknown gearbox type: {gearbox_type}")

    module_name = GEARBOX_DISPATCH[gearbox_type]
    module = __import__(module_name)

    print(f"Running optimization:")
    print(f"  Motor       : {motor}")
    print(f"  Gearbox     : {gearbox_type}")
    print(f"  Gear Ratio  : {gear_ratio}")

    total_time = module.run(motor, gear_ratio)

    print("Optimization completed.")
    print("Time taken:", total_time, "sec")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage:")
        print("  python actOpt.py <motor> <gearbox_type> <gear_ratio>")
        sys.exit(1)

    motor = sys.argv[1]
    gearbox_type = sys.argv[2]
    gear_ratio = float(sys.argv[3])

    main(motor, gearbox_type, gear_ratio)
