
# Manuscript code and data
<h3 style="color: #06f"> Towards a generalized accessibility measure for transportation equity and efficiency</h3>

**Authors**: [Rajat Verma](https://scholar.google.com/citations?hl=en&user=eUl1nl8AAAAJ), [Shagun Mittal](https://scholar.google.com/citations?user=jSrcbicAAAAJ&hl=en&oi=ao), [Mithun Debnath](https://scholar.google.com/citations?user=BFc5p5QAAAAJ&hl=en&oi=ao), [Dr. Satish V. Ukkusuri](https://scholar.google.com/citations?user=9gmoT80AAAAJ&hl=en)

**Submitted to** *[The Journal of Transport Geography](https://www.sciencedirect.com/journal/journal-of-transport-geography)*.

<img src="Study framework.png" width=900>

## Data description
The file [us_bg_access_data.csv](us_bg_access_data.csv) contains the computed values of accessibility for each block group in the contiguous United States by different modes. Figures for walking and bicycling are available only within the boundaries of metropolitan statistical areas (MSAs).
| Field | Data type | Description |
| -     | -     | -             |
| `fips` | Char(12) | 12-digit FIPS code of the block group (2020). |
| `state` | Char(2) | 2-letter code of the parent US state. |
| `county` | Varchar | Name of the parent county. |
| `msa` | Varchar | Principal city in the parent MSA, if any. |
| `{mode}_{purpose}_{thresh}` | Number | Number of places of type `{purpose}` accessible by `{mode}` within `{thresh}` minutes from this block group. <br> **mode**: `drive`, `bike`, `walk` <br> **purpose**: `work` (commute to work), `poi` (all Points of Interest), `shop` (essential shopping places), `serv` (essential services) <br> **thresh**: travel time threshold (30 or 60 minutes) |
