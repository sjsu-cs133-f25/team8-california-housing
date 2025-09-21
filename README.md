# California Housing Prices Analysis — CS 133
## Data Engineering Artifacts Project

### Team
- Sidharth Krishnaswamy
- Rayhann Kwon
- Arthur Wong
- Andrian Than
- Professor: Jelena Gligorijevic

---

## Project Overview

This project demonstrates an analysis of the **California Housing Prices (1990 Census)** dataset. 

- **Artifact A:** Data Card with comprehensive dataset documentation
- **Artifact B:** Dataset transformation with new columns and missing data handling
- **Artifact C:** Grouping, aggregation, and sorting operations with interpretations
- **Artifact D:** Well-labeled visualizations (histograms and bar charts)
- **Artifact E:** Reproducible analysis in a single Jupyter notebook

## Dataset Information

**Source:** California Housing Prices (1990 Census) 
**Shape:** 20,640 census blocks x 9 original features
**Geographic Coverage:** California, USA
**Data Level:** Census block groups

### Key Features:
- `median_income`: Median household income 
- `housing_median_age`: Median age of houses
- `avg_rooms`: Average rooms per household
- `avg_bedrooms`: Average bedrooms per household
- `population`: Total population in block group
- `avg_occupancy`: Average household size
- `latitude`: Latitude coordinate (degrees)
- `longitude`: Longitude coordinate (degrees)
- `median_house_value`: Median house value (in $100,000s)

---

## Quick Start

### 1. Clone and Setup Environment
```bash
git clone https://github.com/sjsu-cs133-f25/team8-california-housing.git
cd team8-california-housing
python3 -m venv .venv

# Mac/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Analysis
```bash
jupyter notebook california_housing_analysis.ipynb
```

---

## Repository Structure

```
team8-california-housing/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── california_housing_analysis.ipynb   # Main analysis notebook (ALL ARTIFACTS)
├── CLAUDE.md                          # Project configuration
├── .gitignore                         # Git ignore rules
└── data/
    └── download_instruction.md        # Alternative data source info
```

---
## Dependencies

See `requirements.txt` for complete list. Key packages:
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `matplotlib>=3.6.0` - Plotting
- `seaborn>=0.12.0` - Statistical visualization
- `scikit-learn>=1.3.0` - Dataset and utilities
- `jupyter>=1.0.0` - Notebook environment
