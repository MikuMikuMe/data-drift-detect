Creating a data drift detection tool involves several steps: loading data, calculating statistics for current and reference data, comparing these statistics, and reporting any drift detected. I'll provide you with a Python program that covers these aspects. This example uses synthetic data for simplicity, but in a real-world situation, you'll use actual datasets.

We'll utilize several libraries in this program, including `pandas` for data manipulation, `numpy` for numerical operations, and `scipy.stats` for statistical tests.

```python
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp  # Kolmogorov-Smirnov test
from scipy.stats import chisquare  # Chi-squared test for categorical features

class DataDriftDetect:
    def __init__(self, reference_data, current_data):
        self.reference_data = reference_data
        self.current_data = current_data

    def detect_numerical_drift(self, column):
        """
        Detect drift in numerical data using the Kolmogorov-Smirnov test.
        Returns p-value, where a low value (e.g., < 0.05) indicates drift.
        """
        ref_col_data = self.reference_data[column].dropna()
        cur_col_data = self.current_data[column].dropna()
        test_result = ks_2samp(ref_col_data, cur_col_data)
        return test_result.pvalue

    def detect_categorical_drift(self, column):
        """
        Detect drift in categorical data using the Chi-squared test.
        Returns p-value, where a low value (e.g., < 0.05) indicates drift.
        """
        ref_counts = self.reference_data[column].value_counts(normalize=True)
        cur_counts = self.current_data[column].value_counts(normalize=True)
        categories = list(set(ref_counts.index).union(set(cur_counts.index)))

        ref_values = [ref_counts.get(cat, 0) for cat in categories]
        cur_values = [cur_counts.get(cat, 0) for cat in categories]

        _, p_value = chisquare(cur_values, f_exp=ref_values)
        return p_value

    def detect_drift(self):
        """
        Detect drift across all columns in the dataset and report.
        """
        drift_results = {}
        for column in self.reference_data.columns:
            if self.reference_data[column].dtype in [np.float64, np.int64]:  # Numerical column
                p_value = self.detect_numerical_drift(column)
                is_drift = p_value < 0.05
            else:  # Categorical column
                p_value = self.detect_categorical_drift(column)
                is_drift = p_value < 0.05

            drift_results[column] = {
                'p_value': p_value,
                'is_drift': is_drift
            }
        return drift_results

# Error handling example
def load_data(file_path):
    """
    Load data from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError as e:
        print(f"Error: The file {file_path} was not found.")
        raise e
    except pd.errors.ParserError as e:
        print(f"Error: The file {file_path} could not be parsed.")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e
    return data

# Example usage:
if __name__ == "__main__":
    # Sample CSV file paths (ensure these files exist or replace with actual file paths)
    reference_data_path = 'reference_data.csv'
    current_data_path = 'current_data.csv'

    # Load the data
    try:
        reference_data = load_data(reference_data_path)
        current_data = load_data(current_data_path)
    except Exception as e:
        print("Failed to load data:", e)
        exit()

    # Instantiate the drift detection tool
    drift_detector = DataDriftDetect(reference_data, current_data)

    # Detect drift
    results = drift_detector.detect_drift()

    # Print results
    for column, result in results.items():
        print(f"Column: {column}, P-Value: {result['p_value']}, Drift Detected: {result['is_drift']}")
```

### Explanation
- The `DataDriftDetect` class performs data drift detection.
  - `detect_numerical_drift`: Uses the Kolmogorov-Smirnov test to detect drift in numerical columns.
  - `detect_categorical_drift`: Uses the Chi-squared test to detect drift in categorical columns.
  - `detect_drift`: Iterates over columns in the dataset and reports drift results.

- The `load_data` function is used to load CSV data and includes error handling for various exceptions:
  - File not found
  - CSV parsing errors
  - General exceptions

### Notes
- The Kolmogorov-Smirnov test is sensitive to differences in distribution shapes and is suitable for numerical data.
- The Chi-squared test is applied to categorical data for detecting distribution differences.
- P-value threshold for detecting drift is typically set at 0.05, but this can be adjusted based on the sensitivity required.
- This tool is designed for educational purposes. In a production environment, consider performance optimization, especially with large datasets.

Ensure that the file paths in the example correspond to your CSV files, or replace them with the actual paths on your system.