# 💎 Diamond Price Prediction using KNN Regression

## 📌 Project Overview
This project predicts the price of diamonds using the K-Nearest Neighbors (KNN) Regression algorithm.

A Machine Learning model is developed using diamond features such as carat, cut, color, clarity, depth, table, length, width, and height.

The application is deployed using Streamlit, allowing users to enter diamond details and get an estimated price.

## 🎯 Project Objective
- Analyze diamond dataset.
- Perform data preprocessing.
- Build a KNN Regression model.
- Evaluate model performance.
- Deploy the application using Streamlit.

## 📂 Project Structure

```text
Diamond-Price-Prediction/
│
├── app.py
├── train_model.py
├── diamonds.csv
├── description.txt
├── knn_pipeline.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Pickle

## ⚙️ Machine Learning Workflow

1. Dataset Collection
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Categorical Encoding
5. Numerical Feature Scaling
6. Train-Test Split (75:25)
7. KNN Regression Model Building
8. Model Evaluation
9. Model Saving using Pickle
10. Streamlit Deployment

## 🤖 Algorithm Used

**K-Nearest Neighbors (KNN) Regression**

KNN Regression predicts the diamond price based on the nearest similar data points in the training dataset.

## 📊 Model Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## 🚀 Live Application

[Click here to use Diamond Price Prediction App](https://diamond-price-prediction-ug2plnokitvdtrzwplo8y9.streamlit.app/)

## 💻 Run Locally

Clone the repository:

```bash
git clone https://github.com/manasareddy2023-pixel/Diamond-Price-Prediction.git
```

Navigate to the project folder:

```bash
cd Diamond-Price-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 👩‍💻 Author

**Manasa Reddy**

GitHub: [manasareddy2023-pixel](https://github.com/manasareddy2023-pixel)

---

⭐ If you find this project useful, feel free to explore the repository!
