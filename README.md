# BCUAnnualReport2026
Data analysis for BCU Annual Report 2026

Code used for data analysis for 2026 annual report

gis_data_subset.ipynb reads in massachusetts geography datasets (not included here) and combine+subset to the greater boston area and write to the shp/ directory
annrep.ipynb is the script (jupyter notebook) that does the real analysis/processing. reads in data from data/ and writes processed data (csv files for excel or other) to analysis/ and figures to figs/
gis_functions.py includes functions written for use in the main script

Data sources
bike blocking reports: https://loudbicycle.com/bb/map 
crash data: https://experience.arcgis.com/experience/bae68e65908f45e1bcc86fe5f089d266/page/
traffic data: https://www.arcgis.com/apps/dashboards/3491f78e46a545fe8186ad70ebe897aa#n 
