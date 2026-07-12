# Car Price Predictor

A machine learning project that predicts used car resale prices based on car name, company, year, kilometers driven, and fuel type — built with Linear Regression.

---

## Project Structure

```
car-price-predictor/
├── src/
│   ├── 01_data_cleaning.py    # clean raw quikr dataset
│   └── 02_model_training.py   # train and save model
├── data/
│   └── quikr_car.csv          # raw dataset
├── app.py                     # Streamlit web app
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Dataset

- Source: Quikr used car listings
- Features: car name, company, year, kilometers driven, fuel type
- Target: resale price in INR

---

## Data Cleaning

- Removed non-numeric year values
- Removed rows where price was listed as "Ask For Price"
- Stripped commas and units from kms driven column
- Removed rows with missing fuel type
- Trimmed car names to first 3 words for grouping
- Removed price outliers above 6 million INR

---

## Model

- Algorithm: Linear Regression with OneHotEncoding for categorical features
- Pipeline: ColumnTransformer + LinearRegression using sklearn make_pipeline
- Best random state: 2711 (found through experimentation)
- R2 Score: 0.83+

---

## How to Run

```bash
git clone https://github.com/Arifkhan171/car-price-predictor.git
cd car-price-predictor
pip install -r requirements.txt
python src/01_data_cleaning.py
python src/02_model_training.py
streamlit run app.py
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

---

## Author

**Arif Khan**
Final Year CS Student | University of Loralai | ML / Deep Learning / Agentic AI

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/arif-khan-71a711376)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github)](https://github.com/Arifkhan171)
