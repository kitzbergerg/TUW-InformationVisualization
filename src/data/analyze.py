import argparse
from pickletools import long1

import pygrib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    args = parser.parse_args()

    grbs = pygrib.open(args.data_path)

    print("--- Finding all unique parameters in the file ---")
    unique_params = set()

    for grb in grbs:
        unique_params.add((grb.shortName, grb.name))

    print("Found parameters (shortName, name):")
    for param in sorted(list(unique_params)):
        print(f"- ({param[0]}, {param[1]})")

    first_grb = grbs.select(shortName='2t', year=2000, month=1)
    assert len(first_grb) == 1
    first_grb = first_grb[0]

    lats, lons = first_grb.latlons()
    print(f"First grb lat/lon shape: {lats.shape}, {lons.shape}")
    print(f"lats: {lats}")
    print(f"lons: {lons}")
    print(f"values: {first_grb.values}")

    grbs.close()