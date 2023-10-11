#%% Imports
import json
from dash import Dash, dcc, html, Input, Output
import geopandas as gpd
import pandas as pd
import plotly.express as px
# import plotly.graph_objects as go

from mobilkit.umni import *

# %% Constants
D = dict
# SCALES = ('BG', 'Tract', 'County')
# MODES = ('Bike', 'Drive', 'Transit', 'Walk')
SCALES = ('BG', 'TRACT', 'COUNTY')

# %% Create the app
app = Dash('SPR 4711')

# %% Load the data
zones = gpd.read_parquet('../data/zones/zones_2010.parquet').set_crs(CRS_DEG)
zjson = {scale: json.loads(df.set_index('geoid')[['geometry']].to_json())
         for scale, df in zones.groupby('scale')}
# zonejson = json.loads(zones.set_index('geoid').to_json())
ej = pd.read_parquet('../data/ejs/ejs.parquet')

# %% App layout
app.layout = html.Div([
    html.H1('My own web application dashboards with Dash', style=D(
        textAlign='left', fontFamily='Noto Sans', color='purple')),
    dcc.RadioItems(id='scale-selector', value='COUNTY', inline=True,
                   style=D(fontFamily='Arial', fontSize='20px'),
                   options=[D(label=x, value=x) for x in SCALES]),
    html.Div(id='output-container', children=[]),
    html.Br(),
    dcc.Graph(id='map-main')
])

# %% Callbacks
@app.callback(
    Output('map-main', 'figure'),
    Input('scale-selector', 'value'))
def plot_map(scale, zones=zones):
    df = zones.query(f'scale=="{scale}"')
    df = df.assign(area_sqmi=df.aland / 2.59e+6)
    fig = px.choropleth(
        df.assign(area_sqmi=df.aland / 2.59e+6),
        geojson=zjson[scale],
        template='plotly_dark',
        locations='geoid', color='area_sqmi')
    fig.update_geos(scope='usa', fitbounds='geojson', visible=True)
    fig.update_layout(height=800, width=1200, margin=D(l=0, r=0, b=0, t=0))
    return fig

# %% Main
if __name__ == '__main__':
    app.run(debug=True, port=8005, threaded=True, use_reloader=True)
