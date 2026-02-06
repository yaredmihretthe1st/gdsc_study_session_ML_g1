# Ride Price Estimation System 🏎️💰

## Project Overview
This project is an end-to-end machine learning solution to **estimate ride prices** and classify rides as high-cost or low-cost. It focuses on designing a dataset, building regression and classification models, and evaluating their performance. The goal is to practice real-world ML workflow rather than maximizing accuracy.

---

## Dataset Description
The dataset was **synthetically created** with 150+ rows and includes **both numerical and categorical features** relevant to ride pricing. The continuous target variable is `ride_price_ETB` (in Ethiopian Birr). A binary classification target `high_cost` was also created by labeling rides above the median price as 1 (high-cost) and others as 0 (low-cost).

**Features Used:**
| Feature | Type | Description | Justification |
|---------|------|-------------|---------------|
| Distance (km) | Numerical | Trip distance | Longer trips usually cost more |
| Trip Duration (minutes) | Numerical | Time taken for trip | Longer durations may increase price |
| Time of Day | Categorical | Morning, Afternoon, Evening, Night | Peak hours may have higher prices |
| Traffic Level | Categorical | Low, Medium, High | Congestion can affect ride price |
| Weather Condition | Categorical | Sunny, Rainy, Snowy | Bad weather can increase fares |
| Demand Level | Numerical | Estimated rider demand | High demand may lead to surge pricing |

---

## How to Run the Notebook
1. Clone the repository:
```bash
git clone <https://github.com/yaredmihretthe1st/gdsc_study_session_ML_g1.git>
2. Open the Jupyter Notebook in Google Colab or Jupyter Lab:
notebook/ride_price_model.ipynb

