## Context

You are preparing a reusable transformation that standardizes player roster data by converting imperial Height and Weight measurements into metric units, while also extracting a separate Jersey Number and cleaning the player Name. The reusable transformation should be built on one team file and then applied to another team file to confirm it works consistently.

## Requirements

- Input the data.
  - Use two input files:
    - `input_01.csv`
    - `input_02.csv`

- Form two reusable steps that can be applied to automatically convert Height and Weight.
  - Build the reusable steps using the provided build file, and then apply them to the other team’s file to test.

- Convert Height from imperial to metric.
  - Use the `HT` field, which contains feet and inches.
  - Convert to meters using 1 inch = 2.54 cm.
  - Round `Height (m)` to 2 decimal places.

- Convert Weight from imperial to metric.
  - Use the `WT` field, which contains a weight expressed in pounds (lbs) or includes a numeric pounds value.
  - Extract the numeric pounds value from `WT` (using the first available integer value); if no numeric value is present, the converted weight should be null.
  - Convert to kilograms using:  
    - kilograms = pounds × 0.453592
  - Round `Weight (KGs)` to 2 decimal places.

- Clean up the `NAME` field and create a `Jersey Number` field.
  - Create `Jersey Number` by extracting trailing digits from the end of `NAME` (keep as text to preserve any leading zeros).
    - Extract digits using the following rule: starting from the rightmost character of `NAME`, extract all consecutive digits (0-9) moving leftward until a non-digit character is encountered.
    - Preserve leading zeros in the extracted digits (e.g., `00` should remain as `"00"`, not `"0"`).
    - Examples: `"LaMarcus Aldridge12"` → extract `"12"`; `"Keldon Johnson3"` → extract `"3"`; `"Rodions Kurucs00"` → extract `"00"`.
    - If no trailing digits exist (i.e., the rightmost character is not a digit), set `Jersey Number` to an empty string.
  - Update `NAME` to be the remaining leading portion after removing the trailing digits, then trim leading/trailing spaces.
  - Handle cases where punctuation appears between the name and jersey number:
    - After extracting the trailing digits, examine the remaining portion of the original `NAME` (before trimming).
    - If the remaining portion ends with one or more punctuation marks (such as periods, dots, commas, or other punctuation characters), and these punctuation marks are followed by optional spaces before the extracted digits in the original `NAME`, then:
      - Include these punctuation marks (and any spaces between them and the digits) in the `Jersey Number` field along with the extracted digits.
      - Remove these punctuation marks and spaces from the cleaned `NAME` before trimming.
    - For example:
      - Original `NAME`: "C.J.Williams.. 9"
        - After digit extraction: remaining portion is "C.J.Williams.. " (with trailing spaces)
        - The remaining portion ends with ".." (punctuation marks)
        - In the original `NAME`, ".." is followed by a space and then the digit "9"
        - Result: `NAME` = "C.J.Williams" (after trimming), `Jersey Number` = ".. 9"
      - Original `NAME`: "John Smith 12"
        - After digit extraction: remaining portion is "John Smith " (with trailing space)
        - The remaining portion ends with a space (not punctuation)
        - Result: `NAME` = "John Smith" (after trimming), `Jersey Number` = "12"
    - The key rule: punctuation marks that appear immediately before (with or without intervening spaces) the trailing digits in the original `NAME` should be moved to `Jersey Number` along with the digits, and removed from `NAME`.

- Remove unrequired fields within the reusable step(s).
  - The final output must include only the required 8 fields listed in the Output section, in the exact order shown.

- Test your reusable steps on another team’s data using the second file.
  - Produce outputs such that:
    - `output_01.csv` is created from `input_02.csv`.
    - `output_02.csv` is created from `input_01.csv`.

- Output the data.

## Output

- output_01.csv
  - 8 fields:
    - Height (m)
    - Weight (KGs)
    - Jersey Number
    - NAME
    - POS
    - AGE
    - COLLEGE
    - SALARY

- output_02.csv
  - 8 fields:
    - Height (m)
    - Weight (KGs)
    - Jersey Number
    - NAME
    - POS
    - AGE
    - COLLEGE
    - SALARY