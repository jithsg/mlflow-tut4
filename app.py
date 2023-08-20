import os
import pickle
import mlflow
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

temp_dir = "temp_model_folder"
os.makedirs(temp_dir, exist_ok=True)

# Load the California Housing dataset
california = fetch_california_housing()
X = california.data
y = california.target
df = pd.DataFrame(california.data, columns=california.feature_names)
dataset_path = f"{temp_dir}/california_housing.csv"

df.to_csv(dataset_path, index=False)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model and log using mlflow
mlflow.set_experiment("mlflow_demo")

with mlflow.start_run():
    # Create model, train it, and create predictions
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    predictions = lr.predict(X_test)

    # Log model
    mlflow.sklearn.log_model(lr, "model")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

    # Log params
    mlflow.log_param("intercept", lr.intercept_)

    # Log metrics
    mlflow.log_metric("mse", mean_squared_error(y_test, predictions))
    mlflow.log_metric("r2", lr.score(X_test, y_test))
    
    # Log dataset
    mlflow.log_artifact(dataset_path, "dataset")

    # Save the model as a pickle file in the temporary folder
    model_path = f"{temp_dir}/model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(lr, f)
    print(f"Model saved to: {model_path}")

    # Log model as artifact
    mlflow.log_artifact(model_path, "model")
    
    plt.scatter(y_test, predictions)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Scatter Plot of True vs Predicted Values')
    plot_path = f"{temp_dir}/scatter_plot.png"
    plt.savefig(plot_path)
    plt.close()

    # Log scatter plot as artifact
    mlflow.log_artifact(plot_path, "scatter_plot")
