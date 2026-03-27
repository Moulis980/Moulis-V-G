# Agricultural Office Optimization for Crop Cultivation

## Project Overview
This project focuses on identifying optimal locations for agricultural offices in Tamil Nadu, India. By analyzing crop cultivation data (area in hectares) and geographical coordinates, the tool provides data-driven recommendations to ensure offices are placed where they can provide maximum support to farmers.

## Problem Statement
Strategic placement of agricultural offices is crucial for efficient resource distribution and farmer outreach. This project solves the challenge of locating these offices by calculating area-weighted centroids and using clustering algorithms to group cultivation hubs effectively.

## Dataset Requirements
The system expects an Excel file named `Dataset.xlsx` with the following structure:
- **Crop**: Name of the agricultural product (e.g., Banana, Sugarcane).
- **District**: The district in Tamil Nadu.
- **Taluk**: The specific sub-district/administrative division.
- **Area (ha)**: The total area under cultivation for that specific crop and location.

## Methodology

### 1. Data Preprocessing
- Cleans numeric data in the `Area (ha)` column, handling string formatting and commas.
- Filters out records with missing or invalid cultivation data.

### 2. Geocoding & Fallback System
- Uses the `geopy` library to fetch Latitude and Longitude for each Taluk/District pair via the Nominatim API.
- Implements a fallback mechanism using a predefined dictionary of District center coordinates to handle API timeouts or missing data.

### 3. Weighted Centroid Analysis
- Calculates a weighted center for each crop, where locations with larger cultivation areas have a higher influence on the result.
- Recommends the nearest existing cultivation site to this theoretical center.

### 4. KMeans Clustering Optimization
- For crops requiring multiple offices (up to 3), the system uses **KMeans Clustering**.
- The clustering is weighted by cultivation area, ensuring office recommendations are biased toward high-production zones.
- Output includes coverage percentages and specific Taluk/District recommendations for each cluster.

## Setup and Usage

### Prerequisites
```bash
pip install pandas numpy geopy scikit-learn folium openpyxl
```

### Running the Analysis
1. Ensure `Dataset.xlsx` is uploaded to the environment.
2. Run the notebook cells to process the geocoding (this includes a 1-second delay between requests to respect API limits).
3. Call the `optimize_agri_offices("Crop Name")` function to get specific recommendations.

## Example Output
For a crop like 'Banana', the system might return:
- Optimal locations in Erode and Theni.
- Coverage percentage for each office based on the surrounding cultivation area.
