## Context
SuperBytes maintains a complaints log that tracks each customer's issue, the date it was received, its resolution status, and key metadata about the product involved. As of 26th June 2024, any complaints still marked In Progress should be considered resolved on that date so performance metrics can be recalculated consistently.

## Requirements
- Input the data
- For Customer Complaints that are still In Progress, set the Date Resolved to today's date (for the output below this is set to 26th June 2024)
- Create a Days to Resolve field for Complaints which are closed
- Create a Days Open field for Complaints which are In Progress
- Update the Timely Response field to correctly say whether it was Timely or Not, based on if the Date Resolved was less than 14 days
- Create a parameter so a user can choose whether to output In Progress complaints or Closed complaints only
- Output the data

## Output

- output_01.csv
  - 11 fields:
    - Complaint ID
    - Receipt Number
    - Customer ID
    - Date Received
    - Date Resolved
    - Timely Response
    - Response to Consumer
    - Issue Type
    - Product Category
    - Product ID
    - Complaint Description
