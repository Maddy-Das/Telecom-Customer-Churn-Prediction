# Telecom Customer Churn Prediction

## Overview

This project predicts customer churn in the telecommunications sector using machine learning and provides an interactive interface built with **Streamlit**. It enables real-time user input and churn predictions to support proactive customer retention strategies.


---

## Technologies Used

* **Streamlit** – Interactive web app
* **Python 3.8+**
* **Pandas**, **NumPy** – Data processing
* **Scikit-learn** – Model training and evaluation
* **Matplotlib**, **Seaborn** – Data visualization

---

## Installation & Setup

```bash
git clone https://github.com/Maddy-Das/Telecom-Customer-Churn-Prediction.git
cd Telecom-Customer-Churn-Prediction

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirement.txt
# If missing, install:
# pip install pandas numpy scikit-learn streamlit matplotlib seaborn
```

---

## Usage

### 1. Preprocess the Data

```bash
python src/preprocess.py
```

Processes data: handles missing values, encodes categorical data, scales numerical features, and optionally outputs `filtered_dataset.csv`.

### 2. Train Models

```bash
python src/train.py
```

Trains predictive models (e.g., Logistic Regression, Random Forest) and saves the best model in `models/`.

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

Opens an interactive UI where users can input customer attributes—like tenure, contract type, monthly charges, and services—and receive real-time churn predictions.

### 4. Prediction Output

Depending on your app logic, it might display:

* **Yes** / **No** (will churn / will stay)
* Probability score of churn
* Optional: Feature importance or explanation (e.g., via SHAP)

---

## Key Features

* Real-time churn prediction through user-friendly forms
* Interactive dropdowns/sliders for customer attributes
* Instant feedback and insights in visual format
* Optionally enhanced with model explainability tools

---

## Contact

* GitHub: [Maddy‑Das](https://github.com/Maddy-Das)
* LinkedIn: [www.linkedin.com/in/maddydas07](https://www.linkedin.com/in/maddydas07)

---
