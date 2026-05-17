# Q3 Data Sources

This directory contains the external data used by `q3_venue_v2.py`.

- `坐标/`: place coordinates in JSONL format. Each line contains `name` and
  `gdm = [longitude, latitude]`. Coordinates were collected with the Gaode
  Maps API and normalized to the 64 participating cities/counties.
- `交通/travel_time_matrix_minutes.csv`: road travel time matrix in minutes.
- `交通/railway_travel_time_matrix_minutes.csv`: railway/public-transport
  travel time matrix in minutes. Counties without direct rail service use a
  nearest-station plus road-transfer estimate.
- `常住人口/常住人口（万人）.json`: resident population in ten-thousand people.
- `可支配收入/可支配收入.json`: disposable income indicators used to form the
  influence Top20 candidate set together with population.

Road, rail, island, and cross-bay records were manually checked where API
queries needed ferry or bridge adjustments. The paper uses these data only as
scenario inputs for sensitivity analysis; all optimization results are
conditioned on the matrices stored here.
