from dash import Dash, dcc, html, Input, Output

D = dict
SCALES = ('BG', 'TRACT', 'COUNTY')

layout = html.Div([
    html.Div(className='header-bar', children=[
        html.Img(src='/assets/indot-logo.png', width=64),
        html.Img(src='/assets/jtrp-logo.png', width=64),
        html.H1('Indiana Transportation Equity Atlas', className='header-title'),
    ]),
    dcc.RadioItems(id='scale-selector', className='radio',
                   value='TRACT', inline=True,
                   options=[D(label=x, value=x) for x in SCALES]),
    html.Div(id='output-container', children=[]),
    html.Br(),
    html.Div(id='map-container-left', className='map-container', children=[
        html.H2('Map header', id='map-header-left-title', className='map-header'),
        dcc.Graph(id='map-left'),
    ])
    # dcc.Graph(id='map-right'),
])