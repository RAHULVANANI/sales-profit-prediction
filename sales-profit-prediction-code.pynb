import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the Dataset

file_path = "sales_data_10k.csv"  # Update path if needed
df = pd.read_csv(file_path)
 
# 1. Heatmap - Correlation Matrix 
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=["number"])

# Plot Heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()
 
# 2. Box Plot - Sales Distribution by Region 
plt.figure(figsize=(8, 6))
sns.boxplot(x="Region", y="Sales ($)", data=df, palette="pastel")
plt.title("Sales Distribution by Region")
plt.xlabel("Region")
plt.ylabel("Sales ($)")
plt.xticks(rotation=45)
plt.show()
 
# 3. Histogram - Distribution of Profit 
plt.figure(figsize=(8, 5))
sns.histplot(df["Profit ($)"], bins=30, kde=True, color="skyblue")
plt.title("Distribution of Profit ($)")
plt.xlabel("Profit ($)")
plt.ylabel("Frequency")
plt.show()
 
# 4. Stacked Bar Chart - Sales and Profit by Category 
# Group sales and profit by category
category_sales_profit = df.groupby("Category")[["Sales ($)", "Profit ($)"]].sum().reset_index()

# Plot stacked bar chart
category_sales_profit.plot(kind="bar", x="Category", stacked=True, figsize=(10, 6), color=["skyblue", "lightcoral"])
plt.title("Sales and Profit by Category")
plt.xlabel("Category")
plt.ylabel("Total ($)")
plt.xticks(rotation=45)
plt.legend(["Sales ($)", "Profit ($)"])
plt.show()
 
# Data Preprocessing for ML Models 
# Drop rows with missing values
df.dropna(inplace=True)

# Check for categorical columns before dummy conversion
categorical_cols = df.select_dtypes(include=["object"]).columns
print(f"Categorical columns: {categorical_cols}")

# Convert categorical variables to dummy variables
df = pd.get_dummies(df, drop_first=True)

# Ensure target column exists and is numeric
if "Profit ($)" not in df.columns or not pd.api.types.is_numeric_dtype(df["Profit ($)"]):
    raise ValueError("Column 'Profit ($)' is either missing or not numeric. Check your dataset.")

# Define Features and Target
X = df.drop(["Profit ($)"], axis=1)  # Features
y = df["Profit ($)"]  # Target variable (predicting Profit)

# Drop non-numeric columns from X
X = X.select_dtypes(include=["number"])

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Model Training and Testing 

# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} Performance:")
    print(f"  Mean Squared Error: {mse:.2f}")
    print(f"  R-squared: {r2:.2f}\n")
    return y_pred
 
# 1. Linear Regression 
lr_model = LinearRegression()
lr_pred = evaluate_model(lr_model, X_train, X_test, y_train, y_test, "Linear Regression")
 
# 2. Random Forest Regressor 
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_pred = evaluate_model(rf_model, X_train, X_test, y_train, y_test, "Random Forest Regressor")
 
# 3. Gradient Boosting Regressor 
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_pred = evaluate_model(gb_model, X_train, X_test, y_train, y_test, "Gradient Boosting Regressor")
 
# 4. Decision Tree Regressor 
dt_model = DecisionTreeRegressor(random_state=42)
dt_pred = evaluate_model(dt_model, X_train, X_test, y_train, y_test, "Decision Tree Regressor")
 
# Visualization of Model Results 
model_results = pd.DataFrame({
    "Actual": y_test,
    "Linear Regression": lr_pred,
    "Random Forest": rf_pred,
    "Gradient Boosting": gb_pred,
    "Decision Tree": dt_pred
})

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
models = ["Linear Regression", "Random Forest", "Gradient Boosting", "Decision Tree"]

# Plot predictions for each model
for model in models:
    sns.scatterplot(x=model_results["Actual"], y=model_results[model], label=model, alpha=0.5)

# Plot perfect prediction line
plt.plot([model_results["Actual"].min(), model_results["Actual"].max()], 
         [model_results["Actual"].min(), model_results["Actual"].max()], 
         color="red", linestyle="--", label="Perfect Prediction")

plt.title("Actual vs Predicted Profit ($)")
plt.xlabel("Actual Profit ($)")
plt.ylabel("Predicted Profit ($)")
plt.legend()
plt.show()
