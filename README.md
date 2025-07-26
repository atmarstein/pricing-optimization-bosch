# Pricing Optimization Model (AWS Deployment)

## Author

Maruf Ajimati
Program: Master of Science in Business Analytics – Nexford University

This repository contains the final project for **BAN6800: Business Analytics Capstone** at **Nexford University**.  
The project focuses on **Pricing Optimization** using machine learning to predict **optimal product prices** and **expected revenue** based on historical retail data and competitor price deltas.

The trained model is deployed on **AWS Elastic Beanstalk** and provides a REST API endpoint for real-time predictions.

---

## **Project Overview**
- **Course:** BAN6800 – Business Analytics Capstone
- **Author:** Maruf Ajimati
- **Institution:** Nexford University
- **Project Goal:** Build and deploy a machine learning model to optimize pricing strategies for a retail dataset, enabling better revenue forecasts and data-driven decisions.
- **Tech Stack:** Python, Flask, Gunicorn, scikit-learn, AWS Elastic Beanstalk.

---

## **Features**
- Machine learning model trained on historical retail price data.
- REST API with `/predict` endpoint for real-time predictions.
- AWS Elastic Beanstalk deployment (scalable cloud hosting).
- Outputs:
  - **Predicted Quantity** (units likely to sell).
  - **Expected Revenue** (forecasted earnings).
  - **Price Recommendation** based on demand and competitor deltas.

---

## **Project Structure**
- `app.py` – Flask app to handle API requests.
- `application.py` – Entry point for AWS Elastic Beanstalk.
- `best_pricing_model.pkl` – Serialized trained ML model.
- `model_feature_columns.txt` – Feature columns required for predictions.
- `pricing_optimization_pipeline.py` – Preprocessing and prediction pipeline.
- `Procfile` – Gunicorn process configuration for deployment.
- `requirements.txt` – Python dependencies.
- `retail_price.csv` – Original dataset.
- `cleaned_retail_price.csv` – Preprocessed dataset.
- `model_performance.csv` & images – Model evaluation metrics.
- `optimal_prices_by_product.csv` – Predicted optimal prices for each product.

---

## **Deployed Model**
The model is deployed on AWS Elastic Beanstalk:  
**[Pricing Optimization Model API](http://pricing-optimization-app-env.eba-xbt3icd4.ap-southeast-2.elasticbeanstalk.com/)**

### **How to Test Predictions**
To use the `/predict` endpoint, send a **POST** request with a JSON payload like:

```json
{
  "unit_price": 100.0,
  "delta_comp_1": 0.1,
  "delta_comp_2": 0.2,
  "delta_comp_3": 0.05
}

License
This repository is for academic purposes only, submitted as part of the Nexford University Capstone Project.
