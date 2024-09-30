import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import matplotlib.pyplot as plt

class MultipleLinearRegression:
    def __init__(self, df, target_column):
        """
        Initialize the class with the dataset and the target column.
        
        Parameters:
        df: pandas DataFrame - the input dataset
        target_column: str - the name of the dependent variable (target)
        """
        self.df = df
        self.target_column = target_column
        self.X = df.drop(columns=[target_column])
        self.y = df[target_column]
        self.model = None
        self.scaler = StandardScaler()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.y_pred = None

    def preprocess_data(self, scale=True):
        """Preprocess the data: Handle missing values, scaling, and splitting into train-test."""
        # Handle missing values (e.g., filling with mean or median if necessary)
        self.df = self.df.fillna(self.df.mean())

        # Split the data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)

        if scale:
            # Scale the features
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)

    def build_model(self):
        """Build and train the linear regression model."""
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        print("Model trained successfully!")

    def evaluate_model(self):
        """Evaluate the model using metrics like R-squared, RMSE, and MAE."""
        # Predicting the test set results
        self.y_pred = self.model.predict(self.X_test)

        # R-squared
        r_squared = self.model.score(self.X_test, self.y_test)
        print(f"R-squared on test data: {r_squared:.4f}")

        # Calculate RMSE and MAE
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        mae = mean_absolute_error(self.y_test, self.y_pred)

        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
    
    def residual_analysis(self):
        """Perform residual analysis by plotting residuals."""
        residuals = self.y_test - self.y_pred
        plt.scatter(self.y_pred, residuals)
        plt.title('Residuals vs Predicted')
        plt.xlabel('Predicted Prices')
        plt.ylabel('Residuals')
        plt.axhline(0, color='red', linestyle='--')
        plt.show()

    def vif_analysis(self):
        """Check for multicollinearity using the Variance Inflation Factor (VIF)."""
        X_scaled = self.scaler.fit_transform(self.X)  # Scale the entire dataset for VIF
        vif = pd.DataFrame()
        vif['Features'] = self.X.columns
        vif['VIF'] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
        print(vif)
    
    def cross_validation(self, k=5):
        """Perform k-fold cross-validation."""
        cv_scores = cross_val_score(self.model, self.scaler.fit_transform(self.X), self.y, cv=k, scoring='neg_mean_squared_error')
        rmse_cv = np.sqrt(-cv_scores.mean())
        print(f"Average RMSE across {k}-fold cross-validation: {rmse_cv:.2f}")
    
    def predict_new(self, new_data):
        """Predict the target value for new data."""
        new_data_scaled = self.scaler.transform([new_data])
        predicted_value = self.model.predict(new_data_scaled)
        return predicted_value[0]
