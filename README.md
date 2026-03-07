# Customer Behaviour Analysis Using K-Means Clustering

## Overview
Analyzed customer behaviour changes over time using transactional data.  
Instead of static spending, the project focuses on **delta (change) features** to capture shifts in activity and segment customers into meaningful groups using **K-Means clustering**.

## Problem
Understanding how spending patterns evolve helps businesses identify **high-value, at-risk, or growing customers**.  
Static clustering often misses these behavioural shifts, limiting actionable insights.

## Dataset
- **Source**: [Online Retail II Dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii)  
- Includes customer transactions, invoice data, product details, and quantities for multiple time periods.

## Tools & Libraries
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## Approach & Methodology
- **Feature Engineering**
  - Calculated **delta features** to measure changes in spending across two periods.
- **Optimal K Selection**
  - Determined the best number of clusters using the **silhouette score**.
- **Clustering**
  - Applied **K-Means clustering** to group customers based on behavioural shifts.
- **Analysis**
  - Interpreted cluster characteristics to derive actionable business insights.

## Results
- Identified **distinct customer segments** based on spending changes.
- Captured **behavioural trends not visible** in raw transactional data.
- Enables **targeted marketing, retention strategies, and personalized campaigns**.
