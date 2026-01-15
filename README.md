# ✈️ Bird Strike Prediction System

An end-to-end **machine learning application** that predicts the **risk of bird strikes during aircraft departure**, along with the **most likely bird species** and **flight phase**, using historical aviation safety data.

The project demonstrates the complete ML lifecycle — from data analysis and feature engineering to model training, evaluation, and deployment using **Streamlit**.

---

## 🚀 Live Application
The model is deployed as an interactive **Streamlit web application** where users can query the system with flight details and receive real-time predictions.

---

## 🎯 Problem Statement
Bird strikes pose a serious safety and operational risk in aviation.  
This project aims to:

- Estimate the **probability of a bird strike occurring**
- Predict the **most likely bird species involved**
- Identify the **most probable flight phase** during which a strike may occur

Predictions are made based on **user-provided flight conditions**.

---

## 🧠 Machine Learning Approach

### Models Used
- **XGBoost (Binary Classification)** – Bird strike probability
- **XGBoost (Multiclass Classification)** – Bird species prediction
- **XGBoost (Multiclass Classification)** – Flight phase prediction

### Why XGBoost?
- Strong performance on tabular data
- Handles nonlinear feature interactions
- Robust to class imbalance
- Widely used in industry

---

## 📊 Model Evaluation

### Bird Strike Probability Model
Evaluated using:
- **ROC-AUC**
- **Log Loss**
- **RMSE on predicted probabilities**

### Species & Flight Phase Models
Evaluated using:
- **Accuracy**
- **Weighted F1-score**
- **Confusion matrices**

Rare bird species were grouped into an **"Other"** category to ensure statistical reliability.

---

## 🖥️ Streamlit Application Features

### User Inputs
- Date of flight
- Time of day (Dawn / Day / Dusk / Night)
- Departure airport
- Visibility conditions

### Model Outputs
- **Bird strike probability (departure phase)**
- **Most likely bird species**
- **Most likely flight phase**

The interface is intentionally minimal and focused on **model querying**.

---

## 🗂️ Project Structure

To run this project locally first clone the git repo and then install the requirements.txt file
after the requirementts are installed run "streamlit run app.py"
