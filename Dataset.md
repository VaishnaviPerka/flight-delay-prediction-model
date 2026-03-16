# Dataset Information

## About the Dataset

This project uses the **U.S. Bureau of Transportation Statistics (BTS) On-Time Performance Database**, which contains detailed information about U.S. domestic flight schedules, operations, and delay indicators.

The full dataset is not included in this repository due to its large size (several million records).

---

## How to Download

1. Go to the BTS On-Time Performance portal:
   👉 https://www.transtats.bts.gov/Tables.asp?QO_VQ=EFD

2. Select the following fields for download:
   - **Flight Date, Year, Quarter, Month, DayOfWeek**
   - **Airline, Origin, Dest**
   - **CRSDepTime, CRSElapsedTime, AirTime, Distance**
   - **DepDel15** (target variable)
   - **DepTimeBlk, Cancelled, Diverted**

3. Download as `.csv` and save locally.

4. Update the file path in `Final_R_code.R`:
   ```r
   path <- "your/local/path/to/dataset.csv"
   ```

---

## Sample Data

A sample of 1,000 rows is provided in `sample_dataset.csv` to illustrate the dataset structure and column names used in this project.

---

## Preprocessing Summary

After downloading, the full pipeline in `Final_R_code.R` will:
- Filter to the top 5 busiest origin airports
- Drop 40+ irrelevant columns
- Handle missing values, encode categoricals, and apply transformations
- Produce a clean training-ready dataset of ~155,945 observations with 14–18 predictors
