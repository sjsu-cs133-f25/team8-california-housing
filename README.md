# California Housing Prices — CS 133

## Team
- Sidharth Krishnaswamy (@) (@)
- Rayhann Kwon (@)
- Arthur Wong (@)
- Andrian Than (@andrianthan)
- Professor: Jelena Gligorijevic

## Dataset
**Title:** California Housing Prices (1990 Census)  
**Summary:** This project uses the **California Housing Prices (1990 Census Data)** dataset.  
It contains information on median house values, population, income, and other features for California census tracts.  
**Main columns:** longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, median_house_value, ocean_proximity.
**Features include:**
- `longitude`: How far west (negative is farther west)
- `latitude`: How far north
- `housing_median_age`: Median age of houses in the district
- `total_rooms`: Total rooms per district
- `total_bedrooms`: Total bedrooms per district
- `population`: Population per district
- `households`: Number of households
- `median_income`: Median income in $10,000s
- `median_house_value`: Median house value in USD
- `ocean_proximity`: Location relative to the ocean

## Repo Layout
- `README.md` — this file
- `data/download_instructions.md` — how to fetch the dataset
- `.gitignore` — prevents data and temp files from being committed

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/sjsu-cs133-f25/team8-california-housing.git
cd team8-california-housing
```
### 2. Create a Virtual Environment
```bash

python3 -m venv .venv

# activate it

# Mac/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```
### 3. Install Dependeniesnstall Dependencies
```bash
pip install -r requirements.txt
```

### 4.  Download the Dataset
```bash
pip install kaggle


# download dataset into data/ folder
kaggle datasets download -d camnugent/california-housing-prices -p data

# unzip and clean up
unzip data/california-housing-prices.zip -d data
rm data/california-housing-prices.zip
```

### 5. Add to .gitignore (we don't want to commit data files)
```bash
# Data files
data/*.csv
data/*.zip

# Python cache
__pycache__/
*.pyc

# Jupyter notebooks
.ipynb_checkpoints/

# OS cruft
.DS_Store
```

### 6. Verify .gitignore status
```bash
git status
```
