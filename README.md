## Download data

Got to https://cds.climate.copernicus.eu/profile to get the api key.

```sh
echo 'url: https://cds.climate.copernicus.eu/api
key: <api-key>' > ~/.cdsapirc

mkdir data
python src/data/download.py
```

Download shapes:

```sh
cd data
mkdir natural_earth_110m
cd natural_earth_110m
wget https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip
unzip ne_110m_admin_0_countries.zip

cd ..
cd ..
```

## Preprocessing

```shell
python src/data/preprocess.py data/data.grib
```

## View

Open `index.html` in a browser (from the root directory) to view the data.
