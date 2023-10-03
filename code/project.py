#%% Imports
import warnings
from pandas import DataFrame as Pdf

warnings.simplefilter(action='ignore', category=FutureWarning)

#%% Constants
# analysis zone scales
SCALES = ('COUNTY', 'TRACT', 'BG')
# travel modes considered, along with their other information
MODES = Pdf(
    columns=['mode', 'gdm_key', 'max_speed', 'color'], data=[
        ['DRIVE', 'driving', 70, 'deepskyblue'],
        ['TRANSIT', 'transit', 20, 'deeppink'],
        ['BIKE', 'bicycling', 16, 'tomato'],
        ['WALK', 'walking', 3.1, 'seagreen'],
]).set_index('mode')
# travel time thresholds (minutes)
MAX_TT = (15, 30, 45, 60)
