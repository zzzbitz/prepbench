## Context
Formula 1 qualifying consists of three rounds: Q1 (18 minutes, eliminate 5 slowest to positions 16-20), Q2 (15 minutes, eliminate 5 slowest to positions 11-15), Q3 (12 minutes, determine positions 1-10). Session transitions can be identified by larger gaps in start times. When calculating session end times, add an extra 2 minutes to catch all drivers.

## Requirements
- Input the data
- Clean the date_start field to make it usable as a date
- Identify Q1 times:
  - Identify the earliest date_start of the session (shortly after Q1 began)
  - Calculate when Q1 ended using the guidelines above (add 2 mins extra)
  - Filter to keep only Q1 times
  - Identify the quickest lap time per driver
  - Rank the drivers AND filter to keep only the 5 slowest (P16-P20)
- Identify Q2 times:
  - Filter out Q1 times
  - Identify the earliest date_start of the session
  - Calculate when Q2 ended using the guidelines above (add 2 mins extra)
  - Use a calculated field to identify Q2 times and Q3 times
  - Filter to keep only Q2 times
  - Identify the quickest lap time per driver
  - Rank the drivers AND filter the 5 slowest (P11-P15)
- Identify Q3 times:
  - From the calculated field where you identified Q3 times, filter out the Q2 times
  - Identify the quickest lap time per driver
  - Rank the drivers (P1-P10)
- Union Q1, Q2 and Q3 together
- Join to your driver data
- Sort to get the correct order for the output
- Output the data

## Output

- output_01.csv
  - 6 fields:
    - Position
    - driver_number
    - driver_code
    - driver_name
    - constructor_sponsor_name
    - lap_duration
