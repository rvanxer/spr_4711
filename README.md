# Spatial Accessibility of America

<!-- # Manuscript code and data -->
<!-- <h3 style="color: #06f"> Towards a generalized accessibility measure for transportation equity and efficiency</h3> -->

**Contributors**: [Rajat Verma](https://scholar.google.com/citations?hl=en&user=eUl1nl8AAAAJ), [Shagun Mittal](https://scholar.google.com/citations?user=jSrcbicAAAAJ&hl=en&oi=ao), [Mithun Debnath](https://scholar.google.com/citations?user=BFc5p5QAAAAJ&hl=en&oi=ao), [Dr. Satish V. Ukkusuri](https://scholar.google.com/citations?user=9gmoT80AAAAJ&hl=en)

Spatial accessibility is defined as the "ease of reaching places from a specific place given travel constraints".
This is a repository of ongoing research in computing different measures of spatial accessibility for the largest cities of the United States (US). The process is explained in detailed in a [manuscript in preparation](https://purdue0-my.sharepoint.com/:w:/r/personal/verma99_purdue_edu/Documents/Purdue/Projects/SPR%204711/Accessibility/JTRG/Accessibility%20paper.docx?d=wddf28f7180bd448b942ba7f1c8c87a8f&csf=1&web=1&e=ojz5uz).

<img src="SAA Framework.png" width=900>

## Data description
The accessibility data for the [50 most populous]((https://en.wikipedia.org/wiki/Metropolitan_statistical_area#Rankings)) Metropolitan Statistical Areas (MSAs) can be downloaded from the following repository:

- **Download link**: [Spatial Accessibility of America (SAA)](https://purdue0-my.sharepoint.com/:f:/g/personal/verma99_purdue_edu/El7mb19l-NFAvukJbig9pLIBB65HegJbb3DdeBNq-P_fpQ?e=oX6z7C) (access code: `access_umni`).

The data file is a comma-separated values (CSV) file containing 216 measures of accessibility for neighborhoods of the 50 most populous urban areas, given at three spatial scales of analysis – (i) county, (ii) census tract, and (iii) census block group (BG), which are identified by their [FIPS](https://en.wikipedia.org/wiki/Federal_Information_Processing_Standards) codes. These are explained below:

| Field | Type | Description |
| -     | -    | -           |
| `FIPS` | Char(12) | FIPS code of the zone based on the 2020 US Census definitions. This is a 12-digit number for BGs, 11-digit for tracts, and 5-digit for counties. |
| `SCALE` | Varchar | Spatial scale of analysis; one of {'County', 'Tract', 'BG'} |
| `STATE` | Char(2) | 2-letter FIPS code of the parent US state. |
| `COUNTY` | Varchar | Name of the parent county. |
| `CITY` | Varchar | Name of the principal city of the MSA. |

The 216 accessibility data columns are coded based on a combination of the measurement model, travel mode, type of amenity, and travel time threshold. Each column denotes the number of opportunities (jobs or POIs) of a given `{kind}` accessible by travel mode `{mode}` within `{thresh}` minutes from the given zone, as measured by the model `{measure}`. These are explained below:
- `{measure}`: Three types of accessibility metrics (for more details, refer this [paper](https://www.researchgate.net/publication/46637359_Accessibility_Measures_Review_and_Applications)):
    - '**C**ontour': Simply counts the total number of opportunities (also known as 'cumulative opportunities').
    - '**G**ravity': Weights opportunities by the time taken to reach them, given by 'impedance functions'. See [this paper](https://conservancy.umn.edu/handle/11299/151329) for info about impedance functions.
    - '**F**loating catchment': Divides the weighted opportunities in the gravity model with the weighted 'reachable population' in each destination zone to account for 'competition' among travelers for the opportunities, such as for jobs or medical facilities where limited seats are available. For more details, see [this paper](https://www.sciencedirect.com/science/article/pii/S0966692301000102?casa_token=w8oNyicGWQYAAAAA:U3pbCKQfheGHYOT0ocDTB7A5O_WnZYtS-pz3j_awynuAtdNgXrZrMUTHK1hLxXXuQQw6GGW_fw). 
- `{mode}`: Three travel modes are covered here:
    - **D**rive: Driving by personal automobile.
    - **W**alk: Walking.
    - **B**ike: Bicycling.
- `{kind}`: Type of opportunity to be accessed:
    - **J0**: All jobs
    - **JW**: Low-wage jobs (≤ $20k per year)
    - **JR**: Jobs for racial minority (non-White) workers
    - **JE**: Jobs for workers with no college degree
    - **P0**: All points of interest (POIs)
    - **PE**: Educational facilities (schools, colleges & universities)
    - **PG**: Grocery stores (including department stores & supercenters)
    - **PM**: Medical ameneties (hospitals, clinics, diagnostic labs, nursing care, etc.)
    - **PS**: Social/community support places (soup kitchens, shelters, day care, retirement, etc.)
- `{thresh}`: Threshold (upper bound) of travel time to be considered for accessibility:
    - **15**, **30**, **45**, and **60** minutes.

>> **Example**: The column `GD30J0` denotes accessibility measured by the `G`ravity model by `D`riving within `30` minutes to `J0` (all jobs). Similarly, `CW15PE` denotes the total number (cumulative opportunities `C`) of educational facilities (`PE`) reachable by `W`alking within `15` minutes.

_Note_: Some combinations are not included here, such as thresholds of 45 and 60 minutes for walking and bicycling, since it is impractical for people to walk or bike for so long to access important opportunities.

## Description of source code files
| File | Description |
| --- | --- |
| [utils.py](code/utils.py) | Utility namespace for common packages, constants & helper functions used in the other files. |
| [spark.py](code/spark.py) | Utility tools for [Pyspark](https://spark.apache.org/docs/latest/api/python/index.html)-based data manipulation. |
| [osrm.py](code/osrm.py) | Script for collecting interzonal impedance using the [OSRM backend](https://github.com/Project-OSRM/osrm-backend) tool. |
| [1_zones.ipynb](code/1_zones.ipynb) | Prepare the geometry layer of the analysis zones for the US at different spatial scales using the [US Census Data API](https://www.census.gov/data/developers/guidance/api-user-guide.html) as well as [OpenStreetMap](https://www.openstreetmap.org) (OSM) database extracts for the MSAs. |
| [2_ses.ipynb](code/2_ses.ipynb) | Process demographic and socio-economic status (SES) data of the analysis zones, mainly from the [American Community Survey](https://www.census.gov/programs-surveys/acs) (ACS). |
| [3_jobs_aaa.ipynb](code/3_jobs_aaa.ipynb) | Extract job counts and flows data from [LEHD LODES](https://lehd.ces.census.gov/data/), along with job accessibility data from the [Access Across America](https://www.cts.umn.edu/programs/ao) (AAA) project (only for comparison). |
| [3_pois.ipynb](code/3_pois.ipynb) | Process Point of Interest (POI) location and visits data from SafeGraph Inc. |
| [4_od_cost.ipynb](code/4_od_cost.ipynb) | Extract and process travel impedance (distance and travel time) data using the [OSRM routing engine](https://github.com/Project-OSRM/osrm-backend). |
| [5_nhts_data.ipynb](code/5_nhts_data.ipynb) | Download and prepare the travel behavior survey data from the [National Household Travel Survey](https://nhts.ornl.gov) (NHTS). |
| [6_nhts_factors.ipynb](code/6_nhts_factors.ipynb) | Identify important factors for use in differentiating impedance functions using the NHTS data prepared in [5_nhts_data.ipynb](code/5_nhts_data.ipynb). |
| [6_get_access.ipynb](code/6_get_access.ipynb) | Compute several measures of accessibility for the study regions using data prepared in the previous files. |
| [7_access_by_imped_funcs.ipynb](code/7_access_by_imped_funcs.ipynb) | Analyze the impact of using a generalized versus case-specific impedance function on accessibility in a case study. |
| [7_analysis_in.ipynb](code/7_analysis_in.ipynb) | Analyze the accessibility variations in Indiana. |
| [7_analysis_us.ipynb](code/7_analysis_us.ipynb) | Analyze the accessibility variations in the biggest MSAs of the US. |

