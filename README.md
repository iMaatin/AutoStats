
# AutoStats<img src='https://raw.githubusercontent.com/iMaatin/AutoStats/main/logo.png' align="right" height="139" />
[![PyPI version](https://img.shields.io/pypi/v/AutoStats.svg)](https://pypi.org/project/AutoStats/)
[![Python versions](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://pypi.org/project/AutoStats/)
[![PyPI - License](https://img.shields.io/pypi/l/AutoStats)](https://opensource.org/licenses/MIT)

AutoStats is a Python library designed to simplify the process of cleaning, imputing, and analyzing datasets with minimal coding effort. It provides tools for generating exploratory reports, handling missing data, and optimizing imputation methods, making it ideal for data scientists and analysts.

---

## 🚀 Features

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

## 📦 Installation

To install AutoStats, ensure you have Python 3.8 or higher and run the following command:

```bash
pip install AutoStats
```

You can also view the project on [PyPI](https://pypi.org/project/AutoStats/).

---

## Usage

### Auto Report
To generate an automated exploratory data analysis report:

```python
from AutoStats.report import auto_report
import pandas as pd

# Load your dataset
df = pd.read_csv("your_dataset.csv")

# Generate the report
auto_report(df, tresh=10, output_file="auto_report.pdf", df_name="Your Dataset")
```

### Manual Report
To create a report with manually specified column types:

```python
from AutoStats.report import manual_report
import pandas as pd

# Load your dataset
df = pd.read_csv("your_dataset.csv")

# Specify column types
categorical_cols = ['col1', 'col2']
continuous_cols = ['col3', 'col4']
discrete_cols = ['col5']

# Generate the report
manual_report(df, categorical_cols, continuous_cols, discrete_cols, output_file="manual_report.pdf", df_name="Your Dataset")
```

### Data Imputation
To run the complete missing data imputation pipeline:

```python
from AutoStats.impute import run_full_pipeline
import pandas as pd

# Load your dataset (with missing values)
df = pd.read_csv("your_dataset.csv")

# Run the imputation pipeline
best_imputed_df, summary_table = run_full_pipeline(df, simulate=True, build=True)

# The pipeline returns the imputed dataframe and a summary of the best methods used.
print(best_imputed_df.head())
print(summary_table)
```

**How the imputation pipeline works:**
- **`simulate=True`**: This mode is for datasets that already have missing values. The pipeline will identify the missing data patterns and find the best imputation method for each column. This is the most common use case.
- **`simulate=False`**: This mode is for evaluating the imputation methods on a complete dataset. You must specify a `missingness_value` (e.g., `missingness_value=0.10` for 10%). The pipeline will artificially introduce missing values into your complete dataset and then impute them, allowing you to assess the performance of the different methods.

For a detailed technical explanation of the imputation module, please refer to the [Imputation Technical Report](impute.pdf).

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

