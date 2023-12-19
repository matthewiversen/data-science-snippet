# Data Science & Analysis Functions Package

This repository contains a collection of functions frequently used in data cleaning, analysis, modeling, and visualization. It's designed to streamline the process for data scientists and analysts by providing reusable code snippets.

## Modules

- analysis.py: Analysis and statistical functions.
- cleaning.py: Data cleaning utilities.
- cols_info.py: Functions to extract information from columns.
- data_preprocessing.py: Preprocessing tools.
- data_statistics.py: Statistical analysis functions.
- logging_errors.py: Error logging mechanisms.
- metrics.py: Metrics calculation for model evaluation.
- model_specific.py: Functions specific to certain models.
- validation.py: Data validation scripts.

## Installation

```bash
pip install zoomds
```

## Import

```python
from zoomds import analysis, cleaning
# or
from zoomds import *
```

## Usage

```python
# example use
analysis.object_cols_distribution(df)
```

## Contributing

Contributions are welcome! Feel free to submit pull requests or open issues for improvements and new features.
