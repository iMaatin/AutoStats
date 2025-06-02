
# AutoStats
![Autostats Banner](logo.png)

AutoStats is a Python library designed to simplify the process of cleaning, imputing, and analyzing datasets with minimal coding effort. It provides tools for generating exploratory reports, handling missing data, and optimizing imputation methods, making it ideal for data scientists and analysts.

---

## Features

### Report Module
- **Auto Report**: Automatically generates an initial exploratory report from your dataset, categorizing columns and visualizing data distributions.
- **Manual Report**: Allows users to specify categorical, continuous, and discrete columns for a more customized report.

### Impute Module
- **Data Preprocessing**: Automatically preprocesses datasets by handling missing values, encoding categorical variables, and identifying column types (categorical, continuous, discrete).
- **Imputation Methods**:
  - **KNN Imputation**: Uses K-Nearest Neighbors to fill missing values.
  - **MICE Imputation**: Implements Multiple Imputation by Chained Equations.
  - **MissForest Imputation**: Uses Random Forests to impute missing values.
  - **MIDAS Imputation**: Leverages deep learning for advanced imputation.
- **Hyperparameter Optimization**: Automatically tunes imputation methods using Optuna for the best performance.
- **Best Method Selection**: Evaluates multiple imputation methods and selects the best-performing one for each column.

---

## Installation

To install AutoStats, ensure you have Python 3.8 or higher and run:

```bash
pip install -r [requirements.txt](http://_vscodecontentref_/0)
```

---

## Usage

To use the Auto Report feature, you can use the following code:

```python
from report import auto_report
import pandas as pd

df = pd.read_csv("your_dataset.csv")
auto_report(df, tresh=10, output_file="auto_report.pdf", df_name="Your Dataset")
```

To use the Manual Report feature, specify your columns as follows:

```python
from report import manual_report

categorical_cols = ['col1', 'col2']
continuous_cols = ['col3', 'col4']
discrete_cols = ['col5']

manual_report(df, categorical_cols, continuous_cols, discrete_cols, output_file="manual_report.pdf", df_name="Your Dataset")
```

To preprocess your data using the `prep` function from the `impute` module, use the following code:

```python
from impute import prep

continuous_cols, discrete_cols, categorical_cols = prep(df)
```

To run the full pipeline with simulation and building options, use the `run_full_pipeline` function:

```python
from impute import run_full_pipeline

best_imputed_df, summary_table = run_full_pipeline(df, simulate=True, build=True, missingness_value=10.0)
```
