#%%
"""
A script to download routing (distance/travel time) data for each 
given origin-destination (OD) pair using the Google Distance Matrix 
(GDM) API. In this case, the ODs are the centroids of analysis 
zones at different scales - county, tract, and block group.

See: https://developers.google.com/maps/documentation/distance-matrix
"""
from pqdm.processes import pqdm
import requests
from scipy.spatial import cKDTree

from mobilkit.umni import *

# departure hour, taken at around the morning peak
DEP_HOUR = 8 # 8 AM
# maximum typical modal speed (miles/hour)
MODE_SPEEDS = {
    'bicycling': 16,
    'driving': 70,
    'transit': 20,
    'walking': 3.1
}
# maximum allowed travel time (minutes)
MAX_TT = 60 # 1 hour of max travel time

#%%
def get_departure_time(use_weekday=True):
    """
    Find the closest upcoming weekday or weekend and set its 
    morning peak hour as the departure time for Google routing.
    """
    date = dt.date.today() + dt.timedelta(days=1)
    date = dt.datetime.fromisoformat(str(date))
    is_weekday = date.weekday() not in [5, 6]
    if use_weekday == is_weekday:
        dep_date = date
    elif is_weekday and not use_weekday:
        ndays = (12 - date.weekday()) % 7
        dep_date = date + dt.timedelta(days=ndays)
        return dep_date
    elif not is_weekday and use_weekday:
        dep_date = date + dt.timedelta(days=2)
    dep_time = dep_date + dt.timedelta(hours=DEP_HOUR)
    return dep_time

#%%
def get_eligible_odps(zones, mode, max_speeds=MODE_SPEEDS,
                      max_tt=MAX_TT, workers=8):
    """
    Filter the OD pairs to be sent to the GDM API by exploiting 
    the maximum speed of the given travel mode, filtering only 
    the destinations lying within a circle of maximum reachable 
    distance in the given time from the origin.
    """
    max_dist = (max_speeds[mode] * U.MI2M / 3600) * (max_tt * 60)
    zones = zones.to_crs(CRS_M).set_index('geoid')[['geometry']]
    zones['geometry'] = zones.centroid
    df = mk.geo.gdf2pdf(zones).set_index(zones.index)
    tree = cKDTree(df[[LON, LAT]])
    odp = tree.query_ball_point(df[[LON, LAT]], max_dist, workers=workers)
    odp = sum([[(i, x) for x in p] for i, p in enumerate(odp)], [])
    odp = pd.DataFrame(odp, columns=['src_id', 'trg_id'])
    df = mk.geo.pdf2gdf(df, crs=CRS_M).to_crs(CRS_DEG)
    df = mk.geo.gdf2pdf(df).set_index(df.index)
    src = df.reset_index().rename(columns=lambda x: 'src_' + x)
    trg = df.reset_index().rename(columns=lambda x: 'trg_' + x)
    odp = odp.merge(src, left_on='src_id', right_index=True)
    odp = odp.merge(trg, left_on='trg_id', right_index=True)
    odp = odp.query('src_geoid != trg_geoid').reset_index(drop=True)
    return odp

#%%
def make_request(query, zero_tol=1e-8):
    """
    Generic request for downloading the distance results using the GDM API.
    """
    base_url = 'https://maps.googleapis.com/maps/api/distancematrix'
    gdm_key = 'AIzaSyDDg_t__52nh3HNfF7fj9fQKV4Rif5DWY0'
    url = f'{base_url}/json?units=metric&key={gdm_key}&{query}'
    resp = requests.request('GET', url, headers={}, data={}).json()
    if resp['status'] != 'OK':
        raise ValueError('Bad response with status: ' + resp['status'])
    df = []
    for row in resp['rows']:
        for dest in row['elements']:
            ok = dest['status'] == 'OK'
            dist = dest['distance']['value'] if ok else np.nan
            time = dest['duration']['value'] if ok else np.nan
            df.append(dict(dist=dist, time=time, ok=ok))
    df = pd.DataFrame(df)
    df['speed'] = df['dist'] / (df['time'] + zero_tol)
    return df

#%%
def get_gdm_tt(zones, scale, mode, weekday=True, bidirec=False,
               njobs=24, save=True, overwrite=False):
    """
    Main function to process and save the GDM travel time data.
    """
    assert scale in ['county', 'tract', 'bg'], scale
    assert mode in ['driving', 'walking', 'bicycling', 'transit'], mode
    day_type = 'weekday' if weekday else 'weekend'
    fname = f'{scale}__{mode}__{day_type}.csv'
    outfile = U.mkfile(f'../data/travel_time/{fname}')
    if outfile.exists() and not overwrite:
        return pd.read_csv(outfile)
    zones = zones[zones['scale'].str.lower() == scale]
    odp = get_eligible_odps(zones, mode).drop(columns=['src_id', 'trg_id'])
    if bidirec:
        nodes = set(odp['src_geoid']).union(set(odp['trg_geoid']))
        all_pairs = pd.DataFrame(list(it.combinations(nodes, 2)),
                                 columns=['src_geoid', 'trg_geoid'])
        odp = odp.merge(all_pairs, on=['src_geoid', 'trg_geoid'])
    dep_time = int(get_departure_time(weekday).timestamp())
    queries = (f'origins={r["src_lat"]},{r["src_lon"]}&' +
               f'destinations={r["trg_lat"]},{r["trg_lon"]}&' +
               f'mode={mode}&departure_time={dep_time}'
               for _, r in odp.iterrows())
    try:
        df = pqdm(queries, make_request, n_jobs=njobs, total=len(odp))
        df = [x for x in df if isinstance(x, Pdf)]
        df = pd.concat(df).reset_index(drop=True)
        df = pd.concat([odp[['src_geoid', 'trg_geoid']], df], axis=1)
        if save:
            df.to_csv(outfile, index=False)
    except Exception as e:
        print(f'`{scale}` by `{mode}`: ERROR: {e}')

#%%
if __name__ == '__main__':
    # t=11:50:51 + a lot of time for (driving @ BG @ weekday)
    weekdays = (True, False)
    scales = ('county', 'tract', 'bg')
    modes = ('walking', 'bicycling', 'transit', 'driving')
    pbar = tqdm(list(it.product(scales, modes, weekdays)))
    zones = gpd.read_parquet('../data/zones/in_2010.parquet')
    for i, (scale, mode, weekday) in enumerate(pbar):
        pbar.set_description(f'{i+1}: {scale} x {mode}')
        get_gdm_tt(zones, scale, mode, weekday, njobs=30)
