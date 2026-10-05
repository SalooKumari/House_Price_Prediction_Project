# 🏠 House Price Prediction

A machine-learning project (IBM SkillsBuild Internship) that predicts house prices from property features.
The **backend** is Python (pandas + scikit-learn) and the **frontend** is a Python web app built with **Streamlit**.

## 📌 Project Overview
Given details of a house — square footage, bedrooms, bathrooms, year built, lot size, garage spaces and neighbourhood quality — the model estimates its market price.
The project covers data checking, exploratory data analysis, feature engineering, comparison of six regression models, hyper-parameter tuning, evaluation and an interactive web interface.

**Result:** the final tuned Ridge Regression model reaches **R² ≈ 0.998**, MAE ≈ 8,175 and RMSE ≈ 10,072 on the 20 % hold-out test set.

## 📂 Dataset
* File: `house_price_regression_dataset.csv` (included in this folder)
* Size: 1,000 rows × 8 columns, no missing values
* Source link: _add the link of the dataset page you downloaded it from (e.g. Kaggle) here_
* Features: `Square_Footage`, `Num_Bedrooms`, `Num_Bathrooms`, `Year_Built`, `Lot_Size`, `Garage_Size`, `Neighborhood_Quality`
* Target: `House_Price`

## 🛠️ Technologies Used
| Area | Tools |
|---|---|
| Language | Python 3.9+ |
| Data handling | pandas, NumPy |
| Machine learning | scikit-learn (Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting) |
| Visualisation | Matplotlib, Seaborn |
| Model saving | joblib |
| Frontend | Streamlit |
| IDE / notebook | VS Code, Jupyter Notebook |

## 📁 Folder Structure
```
House_Price_Prediction/
├── YourName_HousePricePrediction.ipynb   # complete code (EDA, training, evaluation)
├── YourName_ProjectReport.docx           # project report
├── app.py                                # Streamlit frontend
├── requirements.txt                      # dependencies
├── README.md
├── house_price_regression_dataset.csv    # dataset
├── models/                               # saved model + metrics (created by the notebook)
└── figures/                              # charts saved by the notebook
```

## ⚙️ Setup
1. Install **Python 3.9 or newer** and **VS Code** (with the *Python* and *Jupyter* extensions).
2. Extract the ZIP and open the folder in VS Code (`File → Open Folder`).
3. Open the terminal in VS Code (`Ctrl + ~`) and create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate          # Windows
   # source venv/bin/activate     # macOS / Linux
   ```
4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ How to Run
**Step 1 – Run the notebook (training + analysis)**
Open `YourName_HousePricePrediction.ipynb` in VS Code, select the `venv` Python kernel and click **Run All**.
All charts and metrics appear inside the notebook, and the trained model is saved to `models/`.

**Step 2 – Launch the web app (frontend)**
```bash
streamlit run app.py
```
The browser opens at `http://localhost:8501`. The app has four pages: **Predict Price**, **Data Explorer**, **Model Performance** and **About**.

> If you run the app before the notebook, or the saved model is incompatible with your scikit-learn version, the app automatically retrains a model from the CSV.

## 🔍 Key Findings
* `Square_Footage` is by far the strongest predictor (correlation ≈ 0.99 with price).
* Simple linear models (Linear / Ridge / Lasso) outperform tree-based models on this dataset because the price is almost linearly related to the size.
* Residuals are small and evenly spread, so the model generalises well.

## 🚀 Future Improvements
Real-world data with location information, more features, deployment on Streamlit Community Cloud, and price-range (confidence interval) estimates.

## 👤 Author
**Your Name** — IBM SkillsBuild Internship
