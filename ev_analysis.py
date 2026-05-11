# ==========================================================
# MARKET SIZE ANALYSIS ON ELECTRIC VEHICLE INDUSTRIES
# ==========================================================

# -----------------------------
# IMPORT LIBRARIES
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

# -----------------------------
# LOAD DATASET
# -----------------------------

df = pd.read_excel("Electric_Vehicle_Population_Data.xlsx")

# -----------------------------
# DISPLAY BASIC INFORMATION
# -----------------------------

print("\n================ FIRST 5 ROWS ================\n")
print(df.head())

print("\n================ DATASET INFO ================\n")
print(df.info())

print("\n================ DATA DESCRIPTION ================\n")
print(df.describe())

print("\n================ COLUMN NAMES ================\n")
print(df.columns)

# ==========================================================
# DATA CLEANING
# ==========================================================

# -----------------------------
# CHECK MISSING VALUES
# -----------------------------

print("\n================ MISSING VALUES ================\n")
print(df.isnull().sum())

# -----------------------------
# REMOVE MISSING VALUES
# -----------------------------

df = df.dropna()

# -----------------------------
# REMOVE DUPLICATES
# -----------------------------

df = df.drop_duplicates()

print("\n================ DATA AFTER CLEANING ================\n")
print(df.info())

# ==========================================================
# EXPLORATORY DATA ANALYSIS
# ==========================================================

# ----------------------------------------------------------
# 1. EV REGISTRATIONS BY MODEL YEAR
# ----------------------------------------------------------

plt.figure(figsize=(12,6))

df['Model Year'].value_counts().sort_index().plot()

plt.title("EV Registrations by Model Year")

plt.xlabel("Model Year")

plt.ylabel("Number of Registrations")

plt.grid()

plt.show()

# ----------------------------------------------------------
# 2. TOP 10 EV MANUFACTURERS
# ----------------------------------------------------------

plt.figure(figsize=(12,6))

df['Make'].value_counts().head(10).plot(kind='bar')

plt.title("Top 10 EV Manufacturers")

plt.xlabel("Manufacturer")

plt.ylabel("Registrations")

plt.xticks(rotation=45)

plt.show()

# ----------------------------------------------------------
# 3. TOP 10 EV MODELS
# ----------------------------------------------------------

plt.figure(figsize=(12,6))

df['Model'].value_counts().head(10).plot(kind='bar')

plt.title("Top 10 EV Models")

plt.xlabel("Vehicle Model")

plt.ylabel("Registrations")

plt.xticks(rotation=45)

plt.show()

# ----------------------------------------------------------
# 4. EV TYPE DISTRIBUTION
# ----------------------------------------------------------

plt.figure(figsize=(8,8))

df['Electric Vehicle Type'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%'
)

plt.title("Distribution of EV Types")

plt.ylabel("")

plt.show()

# ----------------------------------------------------------
# 5. ELECTRIC RANGE DISTRIBUTION
# ----------------------------------------------------------

plt.figure(figsize=(10,6))

sns.histplot(df['Electric Range'], bins=30)

plt.title("Electric Range Distribution")

plt.xlabel("Electric Range")

plt.ylabel("Frequency")

plt.show()

# ----------------------------------------------------------
# 6. ELECTRIC RANGE VS BASE MSRP
# ----------------------------------------------------------

plt.figure(figsize=(10,6))

sns.scatterplot(
    x=df['Electric Range'],
    y=df['Base MSRP']
)

plt.title("Electric Range vs Base MSRP")

plt.xlabel("Electric Range")

plt.ylabel("Base MSRP")

plt.show()

# ----------------------------------------------------------
# 7. CORRELATION MATRIX
# ----------------------------------------------------------

plt.figure(figsize=(12,8))

corr = df.corr(numeric_only=True)

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Matrix")

plt.show()

# ----------------------------------------------------------
# 8. TOP COUNTIES WITH EV REGISTRATIONS
# ----------------------------------------------------------

plt.figure(figsize=(12,6))

df['County'].value_counts().head(10).plot(kind='bar')

plt.title("Top Counties with EV Registrations")

plt.xlabel("County")

plt.ylabel("Registrations")

plt.xticks(rotation=45)

plt.show()

# ----------------------------------------------------------
# 9. TOP CITIES WITH EV REGISTRATIONS
# ----------------------------------------------------------

plt.figure(figsize=(12,6))

df['City'].value_counts().head(10).plot(kind='bar')

plt.title("Top Cities with EV Registrations")

plt.xlabel("City")

plt.ylabel("Registrations")

plt.xticks(rotation=45)

plt.show()

# ----------------------------------------------------------
# 10. CAFV ELIGIBILITY ANALYSIS
# ----------------------------------------------------------

plt.figure(figsize=(10,6))

df['CAFV Eligibility'].value_counts().plot(kind='bar')

plt.title("CAFV Eligibility Distribution")

plt.xlabel("CAFV Eligibility")

plt.ylabel("Count")

plt.xticks(rotation=45)

plt.show()

# ----------------------------------------------------------
# 11. ELECTRIC RANGE BY MODEL YEAR
# ----------------------------------------------------------

plt.figure(figsize=(12,6))

sns.lineplot(
    x='Model Year',
    y='Electric Range',
    data=df
)

plt.title("Electric Range Growth Over Years")

plt.xlabel("Model Year")

plt.ylabel("Electric Range")

plt.show()

# ----------------------------------------------------------
# 12. BASE MSRP DISTRIBUTION
# ----------------------------------------------------------

plt.figure(figsize=(10,6))

sns.histplot(df['Base MSRP'], bins=30)

plt.title("Base MSRP Distribution")

plt.xlabel("Base MSRP")

plt.ylabel("Frequency")

plt.show()

# ----------------------------------------------------------
# 13. BOXPLOT OF ELECTRIC RANGE
# ----------------------------------------------------------

plt.figure(figsize=(10,6))

sns.boxplot(x=df['Electric Range'])

plt.title("Boxplot of Electric Range")

plt.show()

# ----------------------------------------------------------
# 14. BOXPLOT OF BASE MSRP
# ----------------------------------------------------------

plt.figure(figsize=(10,6))

sns.boxplot(x=df['Base MSRP'])

plt.title("Boxplot of Base MSRP")

plt.show()

# ==========================================================
# FORECASTING EV REGISTRATIONS
# ==========================================================

# ----------------------------------------------------------
# PREPARE DATA
# ----------------------------------------------------------

yearly_data = df['Model Year'].value_counts().sort_index()

X = np.array(yearly_data.index).reshape(-1,1)

y = np.array(yearly_data.values)

# ----------------------------------------------------------
# TRAIN LINEAR REGRESSION MODEL
# ----------------------------------------------------------

model = LinearRegression()

model.fit(X, y)

# ----------------------------------------------------------
# FUTURE PREDICTIONS
# ----------------------------------------------------------

future_years = np.array([
    2025,
    2026,
    2027,
    2028,
    2029,
    2030
]).reshape(-1,1)

future_predictions = model.predict(future_years)

print("\n================ FUTURE EV REGISTRATION FORECAST ================\n")

for year, prediction in zip(future_years.flatten(), future_predictions):

    print(f"{year} : {int(prediction)} registrations")

# ----------------------------------------------------------
# FORECAST GRAPH
# ----------------------------------------------------------

plt.figure(figsize=(12,6))

plt.plot(
    yearly_data.index,
    yearly_data.values,
    label='Historical Data',
    marker='o'
)

plt.plot(
    future_years,
    future_predictions,
    linestyle='dashed',
    marker='o',
    label='Forecast'
)

plt.title("Future EV Registration Forecast")

plt.xlabel("Year")

plt.ylabel("Number of EV Registrations")

plt.legend()

plt.grid()

plt.show()

# ==========================================================
# SUMMARY STATISTICS
# ==========================================================

print("\n================ SUMMARY STATISTICS ================\n")

print("Average Electric Range :")

print(df['Electric Range'].mean())

print("\nMaximum Electric Range :")

print(df['Electric Range'].max())

print("\nAverage Base MSRP :")

print(df['Base MSRP'].mean())

print("\nMost Popular EV Make :")

print(df['Make'].mode()[0])

print("\nMost Popular EV Model :")

print(df['Model'].mode()[0])

# ==========================================================
# FINAL CONCLUSION
# ==========================================================

print("\n================ PROJECT CONCLUSION ================\n")

print("""
The EV market has shown rapid growth over recent years.

Tesla dominates the market in terms of registrations.

Battery Electric Vehicles (BEVs) represent the largest
segment of the EV industry.

Electric range capabilities are increasing steadily.

Forecast analysis suggests continued growth in EV adoption
in the coming years.
""")

print("\n================ PROJECT COMPLETED SUCCESSFULLY ================\n")