# Unlocking YouTube Channel Performance Secrets 📈

Analyzes YouTube channel data to predict estimated revenue (USD). Includes preprocessing, feature engineering (engagement, revenue/view), correlation analysis, RandomForest/XGBoost model training, evaluation (RMSE/R2), and feature importance. Model saved with joblib.

---

## Dataset 📊

This project utilizes the `youtube_channel_real_performance_analytics (1).csv` dataset. It contains various metrics related to YouTube video performance, including views, watch time, likes, comments, shares, subscriber gains, estimated revenue, ad revenue details, and video metadata.

---

## Setup ⚙️

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd <repository-name>
    ```
3.  **Install required libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
    ```
    *(Ensure you have Python and pip installed.)*

---

## Usage 🚀

Open and execute the `Unlocking_YouTube_Channel_Performance_Secrets.ipynb` notebook. You can use environments like Jupyter Notebook, JupyterLab, Google Colab, or VS Code with Python extensions. The notebook details the entire analysis process.

---

## Project Workflow 🛠️

1.  **Import Libraries:** Load necessary libraries like pandas, numpy, seaborn, matplotlib, scikit-learn, xgboost, and joblib.
2.  **Load Data:** Read the CSV dataset into a pandas DataFrame.
3.  **Preprocessing:**
    * Handle missing values (`dropna()`).
    * Remove duplicate columns.
    * Convert `Video Publish Time` to datetime objects.
    * Apply Interquartile Range (IQR) method to remove extreme outliers from numerical columns.
4.  **Feature Engineering:**
    * Create `Revenue per View` (`Estimated Revenue (USD)` / `Views`).
    * Create `Engagement Rate` ((`Likes` + `Shares` + `Comments`) / `Views` \* 100).
5.  **Correlation Analysis:**
    * Generate a heatmap to visualize correlations between numerical features.
    * Select features highly correlated (absolute correlation > 0.3) with the target variable (`Estimated Revenue (USD)`).
6.  **Prepare Data for Modeling:**
    * Separate features (`X`) and target (`y`).
    * Scale numerical features using `StandardScaler`.
    * Split the data into training and testing sets (`train_test_split`).
7.  **Train Models:**
    * Train a `RandomForestRegressor`.
    * Train an `XGBRegressor`.
8.  **Evaluate Models:**
    * Calculate Root Mean Squared Error (RMSE) and R-squared (R²) score for both models on the test set.
    * Visualize the R² score comparison using a bar chart. *Note: The R² scores obtained in the notebook were negative, indicating the models performed worse than a simple mean prediction on this small, outlier-cleaned dataset.*
9.  **Feature Importance:**
    * Extract and plot feature importances from the Random Forest model to identify influential factors.
10. **Save Model:**
    * Save the trained Random Forest model (`rf_model`) to a file (`youtube_revenue_predictor.pkl`) using `joblib`.

---

## Results 📝

* The primary goal was to predict `Estimated Revenue (USD)` using regression models.
* Features like `YouTube Ads Revenue`, `Watch Page Ads Revenue`, `AdSense Revenue`, `CPM`, and `Views` showed higher correlation with the target.
* Evaluation metrics (RMSE and R²) were calculated. The negative R² scores suggest challenges in prediction, potentially due to the very small size of the dataset after aggressive outlier removal (only 8 rows remained).
* Feature importance analysis highlighted the key drivers according to the Random Forest model.
* The trained Random Forest model is saved for potential future use.

---

## Author

* Pranay Kudale
