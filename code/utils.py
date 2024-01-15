# Commonly used built-in imports
import datetime as dt
from glob import glob
import itertools as it
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any
import warnings

# Commonly used external imports
import contextily as ctx
import geopandas as gpd
from geopandas import GeoDataFrame as Gdf
from geopandas import GeoSeries
from IPython.display import display, HTML
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import array as Arr
from numpy.typing import ArrayLike
import pandas as pd
from pandas import DataFrame as Pdf
from pandas import Series
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import yaml

#%% Classes
class Dataset:
    def __init__(self, df, target, rng, test_ratio, n=None,
                 wide=True, factors=[], wts_col='TripWeight', seed=1234):
        # df = df.query('Sampled')# if target == 'Duration' else df
        df = df.sample(n, random_state=seed) if isinstance(n, int) else df
        self.X = df.select_dtypes(CAT)
        if type(factors) in [list, set, tuple] and len(factors) > 0:
            self.X = self.X[list(factors)]
        self.X = pd.get_dummies(self.X, prefix_sep='__') if wide else self.X
        self.y = normalize(df[target], *rng)
        self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(
                self.X, self.y, test_size=test_ratio, random_state=seed)
        wts = df[wts_col].rename('Weight')
        self.wtrain = wts.loc[self.Xtrain.index]
        self.wtest = wts.loc[self.Xtest.index]

#%% Aliases
D = dict
CAT = 'category' # pandas Categorical
CRS_DEG = 'EPSG:4326' # geographical CRS (unit: degree)
CRS_M = 'EPSG:3857' # spatial CRS (unit: meter)
INF = np.inf # infinity
BASEMAP = ctx.providers.OpenStreetMap.Mapnik

# Unit conversion factors
M2FT = 3.28084 # meter to feet
FT2M = 1 / M2FT
MI2M = 1609.34  # mile to meter
M2MI = 1 / MI2M
MI2KM = 1.60934  # mile to kilometer
KM2MI = 1 / MI2KM
SQMI2SQM = 2.59e6  # sq. mile to sq. meter
SQM2SQMI = 1 / SQMI2SQM # sq. m. to sq. mi.
MPS2MPH = 2.2369363 # meters per second to miles per hr
MPH2MPS = 1 / MPS2MPH # miles per hr to meters per second

#%% Helper/utility functions
def mkdir(path: str | Path) -> Path:
    """Make a folder if it does not exist."""
    assert isinstance(path, str) or isinstance(path, Path)
    Path(path).mkdir(exist_ok=True, parents=True)
    return Path(path)


def mkfile(path: str | Path) -> Path:
    """Make the base folder of the given path."""
    assert isinstance(path, str) or isinstance(path, Path)
    path = Path(path)
    return mkdir(path.parent) / path.name


def load_pickle(path: str | Path) -> Any:
    """Load a pickled object."""
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def save_pickle(obj: Any, path: str | Path) -> None:
    """Pickle an arbitrary object."""
    with open(mkfile(path), 'wb') as f:
        pickle.dump(obj, f)


def normalize(x: ArrayLike, vmin=None, vmax=None) -> ArrayLike:
    """Normalize an array of values to fit in the range [0, 1]."""
    if isinstance(x, list) or isinstance(x, tuple):
        x = np.array(x)
    vmin = vmin or np.min(x)
    vmax = vmax or np.max(x)
    return (x - vmin) / (vmax - vmin)


def pdf2gdf(df: Pdf, x='lon', y='lat', crs=None) -> Gdf:
    """Convert a pandas DataFrame to a point GeoDataFrame."""
    geom = gpd.points_from_xy(df[x], df[y], crs=crs)
    return Gdf(df, geometry=geom)


def gdf2pdf(df: Gdf, x='lon', y='lat', crs=None) -> Pdf:
    """Convert a point GeoDataFrame to a DataFrame."""
    if isinstance(crs, str) or isinstance(crs, int):
        df = df.to_crs(crs)
    geom = df if isinstance(df, gpd.GeoSeries) else df.geometry
    return Pdf(geom.apply(lambda g: g.coords[0]).tolist(), columns=[x, y])


def pplot(ax=None, fig=None, size=None, dpi=None, title=None, xlab=None,
          ylab=None, xlim=None, ylim=None, titlesize=None, xlabsize=None,
          ylabsize=None, xeng=False, yeng=False, xticks=None, yticks=None,
          xticks_rotate=None, yticks_rotate=None, xlog=False, ylog=False,
          xminor=True, yminor=True, axoff=False, gridcolor=None, framebordercolor=None):
    """Custom matplotlib plotting function template."""
    if isinstance(size, tuple) and fig is None:
        fig, ax = plt.subplots(figsize=size, dpi=dpi)
    ax = ax or plt.gca()
    ax.set_title(title, fontsize=titlesize or mpl.rcParams['axes.titlesize'])
    ax.set_xlabel(xlab, fontsize=xlabsize or mpl.rcParams['axes.labelsize'])
    ax.set_ylabel(ylab, fontsize=ylabsize or mpl.rcParams['axes.labelsize'])
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    if xeng: ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter())
    if yeng: ax.yaxis.set_major_formatter(mpl.ticker.EngFormatter())
    if xlog: ax.set_xscale('log')
    if ylog: ax.set_yscale('log')
    if xticks: ax.set_xticks(xticks)
    if yticks: ax.set_yticks(yticks)
    if xminor: ax.tick_params(which='minor', bottom=True)
    else: ax.tick_params(which='minor', bottom=False)
    if yminor: ax.tick_params(which='minor', left=True)
    else: ax.tick_params(which='minor', left=False)
    if xticks_rotate:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xticks_rotate)
    if yticks_rotate:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=yticks_rotate)
    if axoff: ax.axis('off')
    if gridcolor: ax.grid(color=gridcolor)
    if framebordercolor:
        for s in ['left', 'right', 'top', 'bottom']:
            ax.spines[s].set_color(framebordercolor)
    ax.set_axisbelow(True)
    fig = fig or plt.gcf()
    return ax


def imsave(title=None, fig=None, ax=None, dpi=300,
           root='./fig', ext='png', opaque=True):
    """Custom method to save the current matplotlib figure."""
    fig = fig or plt.gcf()
    ax = ax or fig.axes[0]
    title = title or fig._suptitle or ax.get_title() or 'Untitled {}'.format(
        dt.datetime.now().strftime('%Y-%m-%d_%H-%m-%S'))
    title = re.sub(r'[^A-Za-z\s\d,.-]', '_', title)
    fig.savefig(f'{mkdir(root)}/{title}.{ext}', dpi=dpi, bbox_inches='tight',
                transparent=not opaque, facecolor='white' if opaque else 'auto')


def disp_table(df: Pdf, styles=()) -> None:
    """Fancy display a Pandas dataframe in notebooks with custom styles."""
    display(HTML(df.style.set_table_styles(styles)
                 .to_html().replace('\\n', '<br>')))


def disp(x: Pdf | Gdf | Series | GeoSeries, top=1):
    """Custom display for DataFrame and Series objects in Jupyter notebooks."""
    def f(tabular: bool, crs: bool):
        shape = ('{:,} rows x {:,} cols'.format(*x.shape) if tabular
                 else f'{x.size:,} rows')
        mem = x.memory_usage(deep=True) / (1024 ** 2)
        mem = f'Memory: {(mem.sum() if tabular else mem):.1f} MiB'
        crs = f'CRS: {x.crs.srs}' if crs else ''
        print(shape + '; ' + mem + ('; ' + crs if crs else ''))
        if tabular:
            types = Pdf({x.index.name or '': '<' + x.dtypes.astype(str) + '>'}).T
            display(pd.concat([types, x.head(top).astype({'geometry': str}) 
                               if crs else x.head(top)]))
        else:
            print(x.head(top))
    if isinstance(x, Gdf): f(True, True)
    elif isinstance(x, Pdf): f(True, False)
    elif isinstance(x, GeoSeries): f(False, True)
    elif isinstance(x, Series): f(False, False)
    return x

#%% Settings
# Ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Set the newer pyogrio engine for geopandas for faster operations
gpd.options.io_engine = 'pyogrio'
# default plot settings
plt.rcParams.update({
    'axes.edgecolor': 'k',
    'axes.edgecolor': 'k',
    'axes.formatter.use_mathtext': True,
    'axes.grid': True,
    'axes.labelcolor': 'k',
    'axes.labelsize': 13,
    'axes.linewidth': 0.5,
    'axes.titlesize': 15,
    'figure.dpi': 150,
    'figure.titlesize': 15,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Serif'],
    'grid.alpha': 0.15,
    'grid.color': 'k',
    'grid.linewidth': 0.5,
    'legend.edgecolor': 'none',
    'legend.facecolor': '.9',
    'legend.fontsize': 11,
    'legend.framealpha': 0.5,
    'legend.labelcolor': 'k',
    'legend.title_fontsize': 13,
    'mathtext.fontset': 'cm',
    'text.color': 'k',
    'text.color': 'k',
    # 'text.usetex': True,
    'xtick.bottom': True,
    'xtick.color': 'k',
    'xtick.labelsize': 10,
    'xtick.minor.visible': True,
    'ytick.color': 'k',
    'ytick.labelsize': 10,
    'ytick.left': True,
    'ytick.minor.visible': True,
})
# add the `disp` method to pandas and geopandas series & DF classes
pd.DataFrame.disp = disp
gpd.GeoDataFrame.disp = disp
gpd.GeoSeries.disp = disp
pd.Series.disp = disp
