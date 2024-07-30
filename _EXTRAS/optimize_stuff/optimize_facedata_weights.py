import os
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from joblib import Parallel, delayed
import scipy.stats as stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to read data from directory
def read_data_from_directory(directory):
    all_data = []
    try:
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r') as file:
                    data = file.read().strip('[]').split(',')
                    cleaned_data = [float(value.strip()) for value in data if 0 <= float(value.strip()) <= 1]
                    all_data.extend(cleaned_data)
    except Exception as e:
        logging.error(f"An error occurred while reading {filepath}: {e}")
    return np.array(all_data, dtype=np.float16)

# Function to calculate metrics
def calculate_metrics(values):
    if len(values) == 0:
        return [np.nan] * 6
    median = np.median(values)
    sd = np.std(values)
    iqr = np.percentile(values, 75) - np.percentile(values, 25)
    perc_90 = np.percentile(values, 90)
    skewness = stats.skew(values) * 0.1 * np.power(median, 2)/iqr
    kurt = stats.kurtosis(values) * 0.1 * np.power(median, 2)/iqr
    
    return [median, sd, iqr, perc_90, skewness, kurt]

# Custom objective function for RandomizedSearchCV
def custom_objective(weights, metrics, scores):
    combined_scores = np.dot(metrics, weights)
    performance = np.mean((combined_scores - scores) ** 2)
    return performance

# Custom RandomizedSearchCV class with adaptive stopping
class CustomRandomizedSearchCV:
    def __init__(self, param_distributions, n_iter, metrics, scores, tol=1e-5, patience=10):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.metrics = metrics
        self.scores = scores
        self.tol = tol
        self.patience = patience

    def fit(self):
        best_score = float('inf')
        best_weights = None
        no_improve_count = 0
        
        for i in range(self.n_iter):
            weights = np.random.uniform(self.param_distributions['weights'][0], self.param_distributions['weights'][1], self.metrics.shape[1])
            weights /= np.sum(weights)  # Normalize weights
            score = custom_objective(weights, self.metrics, self.scores)
            
            if score < best_score:
                best_score = score
                best_weights = weights
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= self.patience:
                logging.info(f"Stopping early at iteration {i+1} with best score {best_score:.6f}")
                break

            logging.info(f"Iteration {i+1}/{self.n_iter}, score: {score:.6f}, best score: {best_score:.6f}")

        self.best_weights_ = best_weights
        self.best_score_ = best_score

# Function to optimize weights using Custom Randomized Search
def optimize_weights(metrics, scores, n_iter=100, tol=1e-5, patience=10):
    param_distributions = {'weights': (0, 1)}
    search = CustomRandomizedSearchCV(param_distributions, n_iter, metrics, scores, tol, patience)
    search.fit()
    return search.best_weights_

# Function to evaluate order and weights in parallel
def evaluate_order_and_weights(metrics, scores, n_iter=100, n_jobs=10):
    fixed_order = ["Median", "P90", "Sc_Skewness", "Sc_Kurtosis", "IQR", "SD"]
    metric_names = ["Median", "SD", "IQR", "P90", "Sc_Skewness", "Sc_Kurtosis"]
    
    ordered_metrics = np.array([metrics[:, metric_names.index(name)] for name in fixed_order]).T

    weights = optimize_weights(ordered_metrics, scores, n_iter)
    combined_scores = np.dot(ordered_metrics, weights)
    performance = np.mean((combined_scores - scores) ** 2)

    return fixed_order, weights, performance

# Function to generate theoretical scores
def generate_theoretical_scores(metrics, num_models, fixed_value=0.3):
    try:
        theoretical_scores = np.array([np.median(metrics[:, i]) for i in range(metrics.shape[1])])
        if not np.all(np.isfinite(theoretical_scores)):
            raise ValueError("Non-finite median value found")
        theoretical_scores = np.tile(theoretical_scores, (num_models, 1)).mean(axis=1)
    except Exception as e:
        logging.warning(f"Failed to calculate median values, using fixed value: {e}")
        theoretical_scores = np.full(num_models, fixed_value, dtype=np.float16)
    return theoretical_scores

def main():
    # Assuming this script is placed in the 'output' directory
    base_dir = os.getcwd()
    
    model_dirs = [os.path.join(base_dir, subdir) for subdir in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, subdir))]
    
    if len(model_dirs) < 2:
        logging.error("Error: There must be at least two model directories to compare.")
        return
    
    # Read data and calculate metrics for each model directory
    all_metrics = []
    for model_dir in model_dirs:
        values = read_data_from_directory(model_dir)
        metrics = calculate_metrics(values)
        all_metrics.append(metrics)
    
    all_metrics = np.array(all_metrics, dtype=np.float16)
    
    # Generate theoretical scores based on the median values or a fixed value
    num_models = len(model_dirs)
    theoretical_scores = generate_theoretical_scores(all_metrics, num_models)
    
    # Evaluate the best order and weights in parallel
    best_order, best_weights, best_score = evaluate_order_and_weights(all_metrics, theoretical_scores, n_iter=100, n_jobs=10)

    # Print the results
    logging.info(f"Best Order: {best_order}")
    logging.info(f"Best Weights: {best_weights}")
    logging.info(f"Best Score: {best_score}")

if __name__ == "__main__":
    main()
