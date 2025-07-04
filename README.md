# Crimes in Boston Analysis and Prediction

## Project Overview
This repository contains the code and report for an applied data-mining project that explores crime patterns in Boston and builds machine-learning models to predict the type of crime based on historical data. The analysis covers data preprocessing, exploratory data analysis (EDA), feature engineering, model training (Logistic Regression & Random Forest), and evaluation.

## Repository Structure
```

.
├── data/
│   └── crimes.csv               # Boston crimes dataset (downloaded from Kaggle)
│   └── offense_codes.csv
├── notebook/
│   └── Final\_Project.ipynb      # Jupyter notebook containing all code & visualizations
├── report/
│   └── crimes-in-boston-report.docx.pdf   # Project report&#x20;
├── requirements.txt             # Python dependencies
└── README.md                    # This file

```

## Data
- **Source:** “Crimes in Boston” dataset on Kaggle:  
  https://www.kaggle.com/datasets/AnalyzeBoston/crimes-in-boston  
- **Contents:**  
  - `INCIDENT_NUMBER`, `OFFENSE_CODE`, `OFFENSE_CODE_GROUP`, `OFFENSE_DESCRIPTION`  
  - `DISTRICT`, `REPORTING_AREA`  
  - `SHOOTING` (Y/N), `OCCURRED_ON_DATE` (timestamp)  
  - `YEAR`, `MONTH`, `DAY_OF_WEEK`, `HOUR`  
  - `Lat`, `Long`  

Place the unzipped `crimes.csv` file under the `data/` folder before running the analysis.

## Installation

1. **Clone this repository**  
   ```
   git clone https://github.com/your-username/boston-crime-prediction.git
   cd boston-crime-prediction
 
  ``

2. **Create a Python 3 (≥3.7) virtual environment**

   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```
   pip install -r requirements.txt
   ```

   The key packages are:

   * **PySpark** (for scalable data processing & ML pipelines)
   * **scikit-learn** (KMeans, PCA)
   * **numpy**, **scipy**
   * **matplotlib**, **seaborn**, **plotly** (visualizations)
   * **folium**, **branca** (interactive maps)
   * **IPython** (notebook display utilities)

4. **(Optional)** Install JupyterLab or Jupyter Notebook

   ```
   pip install jupyterlab
   ```

## Usage

1. **Download the dataset** from Kaggle and place `crimes.csv` in `data/`.
2. **Launch Jupyter**
3. **Open and run** `notebooks/Final_Project.ipynb`.

   * The notebook is organized into sections for data loading, preprocessing, EDA, feature engineering, model training, evaluation, and mapping.
   * You can re-run each cell in sequence to reproduce all results and figures.

## Methodology

1. **Data Preprocessing**

   * Handle missing values (mean/median for numerics; ‘N’ for shooting flag).
   * Remove duplicates on `INCIDENT_NUMBER`.
   * Convert `OCCURRED_ON_DATE` to datetime, extract `DayOfYear`, `WeekOfYear`, `Hour`, `DayOfWeek`.
   * Index categorical features (`DISTRICT`, `OFFENSE_CODE_GROUP`) via `StringIndexer`.

2. **Exploratory Data Analysis (EDA)**

   * Descriptive statistics & distributions of crime types, districts, time patterns.
   * Visualizations: bar charts, heat maps, time series plots, geospatial maps with Folium.

3. **Feature Engineering**

   * Numerical encoding of categorical variables.
   * Spatial clustering via K-Means & dimensionality reduction with PCA (for hotspot detection).

4. **Modeling**

   * **Logistic Regression** (Spark ML)
   * **Random Forest Classifier** (Spark ML)
   * Pipeline: VectorAssembler → (StringIndexer) → Model
   * Evaluation metrics: Accuracy, Precision, Recall, F1-Score on train, validation, and test sets.

5. **Feature Importance & Interpretation**

   * Coefficients from Logistic Regression.
   * Gini importance from Random Forest.
   * Discussion of key predictors: district, time features, geolocation.

## Results

* **Logistic Regression** achieved \~67% accuracy on the test set.
* **Random Forest** reached \~31% accuracy but provided richer feature-importance insights.
* Temporal peaks: late nights & weekends; spatial hotspots: certain districts.
* Full results, figures, and discussion are in `reports/crimes-in-boston-report.docx.pdf`.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes, improvements, or new analyses.

## Author

* **Hammad Anwar**
* **Supervisor:** Sir Asif Khalid, SZABIST Karachi

