import argparse
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pygrib
import geopandas as gpd
from shapely import Point
from tqdm import tqdm


def create_country_mapping(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Create a mapping from a lat/lon grid to ISO_A3 country codes.

    Parameters:
    -----------
    lats : np.ndarray
        2D array of latitudes (n_lat, n_lon)
    lons : np.ndarray
        2D array of longitudes (n_lat, n_lon)

    Returns:
    --------
    np.ndarray
        2D array of ISO_A3 country codes. 'XXX' for no country.
    """
    print(f"Creating country mapping for grid shape {lats.shape}...")

    # 1. Load World Shapefile
    shapefile_path = 'data/natural_earth_110m/ne_110m_admin_0_countries.shp'
    if not os.path.exists(shapefile_path):
        print(f"!!! ERROR: Shapefile not found at {shapefile_path}")
        print("Please download the '110m Cultural' Admin 0 countries shapefile from Natural Earth.")
        raise FileNotFoundError(shapefile_path)

    countries_gdf = gpd.read_file(shapefile_path)[['ISO_A3', 'geometry']]

    # 2. Create GeoDataFrame from GRIB points
    lons = np.where(lons > 180, lons - 360, lons)
    points_geom = [Point(lon, lat) for lon, lat in zip(lons.flatten(), lats.flatten())]
    points_gdf = gpd.GeoDataFrame(geometry=points_geom, crs="EPSG:4326")

    # 3. Perform the Spatial Join
    print("Performing spatial join...")
    joined_gdf = gpd.sjoin(points_gdf, countries_gdf, how='left')

    # 4. Extract results and reshape
    iso_codes_flat = joined_gdf['ISO_A3'].fillna('XXX').astype(str)
    iso_map = iso_codes_flat.values.reshape(lats.shape)

    print(f"Found {np.unique(iso_map[iso_map != 'XXX']).size} unique countries.")
    print(f"Grid points: {iso_map.size} ({np.sum(iso_map == 'XXX')} ocean/unmapped)")
    return iso_map


def calculate_country_averages(values: np.ndarray,
                               iso_map: np.ndarray,
                               mask_value: Optional[float] = None) -> Dict[str, float]:
    """
    Calculate average values for each country using a fast, vectorized approach.

    Parameters:
    -----------
    values : np.ndarray
        2D array of values to average, same shape as iso_map.
        Can be a masked array.
    iso_map : np.ndarray
        2D array of ISO 3-letter country codes
    mask_value : float, optional
        Value to treat as masked/invalid (e.g., np.nan)

    Returns:
    --------
    Dict[str, float]
        Dictionary mapping ISO country codes to their average values
    """
    # 1. Get flattened values and create validity mask
    if np.ma.is_masked(values):
        valid_mask_ma = ~values.mask.flatten()
        values_flat = values.data.flatten()
    else:
        valid_mask_ma = np.ones(values.size, dtype=bool)
        values_flat = values.flatten()

    # 2. Add explicit mask_value (e.g., np.nan)
    if mask_value is not None:
        if np.isnan(mask_value):
            valid_mask_ma &= ~np.isnan(values_flat)
        else:
            valid_mask_ma &= (values_flat != mask_value)

    # 3. Create land mask (excluding 'XXX')
    iso_map_flat = iso_map.flatten()
    land_mask = (iso_map_flat != 'XXX')

    # 4. Combine masks
    final_mask = valid_mask_ma & land_mask

    # 5. Create DataFrame *only* from valid, on-land points
    if not np.any(final_mask):
        return {}  # No valid data

    df = pd.DataFrame({
        'country': iso_map_flat[final_mask],
        'value': values_flat[final_mask]
    })

    # 6. Perform the fast groupby-mean
    country_averages = df.groupby('country')['value'].mean()

    return country_averages.to_dict()


# --- Main execution block ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Aggregate GRIB '2t' data by country and month."
    )
    parser.add_argument("data_path", type=str, help="Path to the GRIB file")
    args = parser.parse_args()

    cache_dir = "data"
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Opening GRIB file: {args.data_path}")
    try:
        grbs = pygrib.open(args.data_path)
    except Exception as e:
        print(f"!!! FATAL: Could not open GRIB file. Error: {e}")
        exit(1)

    # --- Step 1: Get or create the country mapping ---
    try:
        first_grb = grbs.select(shortName='2t')[0]
        lats, lons = first_grb.latlons()
    except Exception as e:
        print(f"!!! FATAL: Error reading first '2t' message: {e}")
        print("Cannot determine grid, exiting.")
        grbs.close()
        exit(1)

    # CRITICAL: Cache file MUST be grid-dependent
    grid_shape_str = f"{lats.shape[0]}x{lats.shape[1]}"
    iso_map_file = os.path.join(cache_dir, f'iso_map_{grid_shape_str}.npy')

    if os.path.exists(iso_map_file):
        print(f"Loading cached country map: {iso_map_file}")
        iso_map = np.load(iso_map_file)
    else:
        iso_map = create_country_mapping(lats, lons)
        print(f"Saving country map to {iso_map_file}")
        np.save(iso_map_file, iso_map)

    # --- Step 2: Iterate all '2t' messages and aggregate ---
    temp_grbs = grbs.select(shortName='2t')
    if not temp_grbs:
        print("No '2t' messages found in file. Exiting.")
        grbs.close()
        exit(0)

    all_results = []
    print(f"Processing {len(temp_grbs)} '2t' messages...")

    for grb in tqdm(temp_grbs, desc="Aggregating GRIB data"):
        # Calculate averages for all countries for this timestep
        country_avg_kelvin = calculate_country_averages(grb.values, iso_map)

        # Add results to our list
        for country, avg_k in country_avg_kelvin.items():
            all_results.append({
                'year': grb.year,
                'month': grb.month,
                'country': country,
                'temperature_celsius': avg_k - 273.15  # Convert K to C
            })

    grbs.close()

    # --- Step 3: Create and format the Pandas DataFrame ---
    if not all_results:
        print("No valid data found during processing. Exiting.")
        exit(0)

    print("Creating Pandas DataFrame...")
    df = pd.DataFrame(all_results)
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    df = df[['date', 'country', 'temperature_celsius', 'year', 'month']]
    df = df.set_index('date').sort_values(by=['date', 'country'])

    # --- Display results ---
    print("\n--- Aggregation Complete ---")
    print("\nDataFrame Info:")
    df.info()

    print("\nDataFrame Head:")
    print(df.head())

    # Save to CSV
    output_csv = os.path.join(cache_dir, 'country_monthly_temperatures.csv')
    df.to_csv(output_csv)
    print(f"\nResults saved to {output_csv}")
