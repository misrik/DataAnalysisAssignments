import pandas as pd

def parse_data():

    fpath = 'Earthquakes.csv'
    df0 = pd.read_csv(fpath)

    df = df0[['Death Description', 'Mag', 'Focal Depth (km)', 'Latitude', 'Longitude']]
    df = df.dropna(subset=[
        'Death Description',
        'Mag',
        'Focal Depth (km)',
        'Latitude',
        'Longitude'
    ])

    df = df.rename(columns={
        'Focal Depth (km)': 'Focal Depth',
        'Latitude': 'Lat',
        'Longitude': 'Long'
    })

    print(df.describe(include='all'))
    print()

    return df
