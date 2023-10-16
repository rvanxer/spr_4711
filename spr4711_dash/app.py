#%% Imports
import json
from dash import Dash, dcc, html, Input, Output
import geopandas as gpd
import pandas as pd
import plotly.express as px
# import plotly.graph_objects as go

from mobilkit.umni import *
from layout import layout

# %% Constants
D = dict
IN_CENTER = D(lon=-86.28146, lat=39.92014) # Indiana's centroid
SCALES = ('BG', 'TRACT', 'COUNTY')

# %% Create the app
app = Dash(__name__, title='SPR 4711',
           external_stylesheets=['./assets/stylesheet.css'])

# %% Load the data
zones = gpd.read_parquet('../data/zones/zones_2010.parquet').set_crs(CRS_DEG)
zones.geometry = zones.geometry.simplify(0.007)
geos = {scale: json.loads(df.set_index('geoid')[['geometry']].to_json())
         for scale, df in zones.groupby('scale')}
ej = pd.read_parquet('../data/ejs/ejs.parquet')
ses = (ej.query('is_ses & is_derived & is_pctile')
       [['scale', 'variable', 'geoid', 'value']]
       .reset_index(drop=True))

# %% App layout
app.layout = layout

# %% Callbacks
@app.callback(
    [Output('map-left', 'figure'),
     Output('map-header-left-title', 'children')],
    Input('scale-selector', 'value'))
def plot_map(scale, variable='% low life expectancy', zones=zones, ses=ses):
    df = zones.query(f'scale=="{scale}"')
    df = df.merge(ses, on='geoid').query(f'variable=="{variable}"')
    df = df.assign(area_sqmi=df.aland / 2.59e+6)
    fig = px.choropleth_mapbox(
        df,
        geojson=geos[scale],
        color='area_sqmi' if scale == 'COUNTY' else 'value',
        mapbox_style='carto-positron',
        locations='geoid',
        color_continuous_scale='Blues',
        hover_data=['geoid', 'name', 'area_sqmi'],
        zoom=6.5, center=IN_CENTER)
    fig.update_geos(scope='usa', fitbounds='geojson', visible=True)
    fig.update_layout(height=800, width=1200, margin=D(l=0, r=0, b=0, t=0),
                      hoverlabel=D(bgcolor='white', font_size=18, font_family='Arial'))
    return fig, [variable]

# %% Main
if __name__ == '__main__':
    app.run(debug=True, port=8005, threaded=True, use_reloader=True)
