import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import skew, kurtosis, shapiro
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

# ===== USER DEFINED VARIABLES =============================
# Define the paths to the directories containing the dataset TXT files
dir_model_1 = r"C:\Users\klrom\Desktop\Auto1111\stable-diffusion-webui\outputs\txt2img-images\2024-07-21\data_model_1"
dir_model_2 = r"C:\Users\klrom\Desktop\Auto1111\stable-diffusion-webui\outputs\txt2img-images\2024-07-21\data_model_2"
# ===== USER DEFINED VARIABLES =============================

# ===== LOGGING ============================================
# This section should ensure that output data is displayed
# both in the terminal window as well as an output TXT file
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Remove any existing handlers
if logger.hasHandlers():
    logger.handlers.clear()

# Create a string stream for logging
from io import StringIO
log_stream = StringIO()

# Create a stream handler to write logs to the string stream
stream_handler = logging.StreamHandler(log_stream)
stream_handler.setLevel(logging.INFO)

# Create a console handler to write logs to the terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a logging format and set it for both handlers
formatter = logging.Formatter('%(message)s')
stream_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add both handlers to the logger
logger.addHandler(stream_handler)
logger.addHandler(console_handler)
# ===== LOGGING ============================================

# ===== FUNCTION ZOO =======================================
# Function to read and clean the data from all TXT files in a directory
# Sometimes the face embed-distance analysis generates an outlier
# when it cannot find a face, etc., and this culls those data from the set
def read_data_from_directory(directory):
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r') as file:
                    data = file.read().strip('[]').split(',')
                    cleaned_data = [float(value.strip()) for value in data if 0 <= float(value.strip()) <= 1]
                    all_data.extend(cleaned_data)
            except Exception as e:
                logger.info(f"An error occurred while reading {filepath}: {e}")
                continue
    return np.array(all_data)

# Function to calculate metrics
# I chose as many metrics as I though meaningful, but I'm not an experienced statistician
# To help combat my ignorance, these will be tried in five different methods
def calculate_metrics(values):
    mean = np.mean(values)
    sd = np.std(values)
    med = np.median(values)
    iqr = np.percentile(values, 75) - np.percentile(values, 25)
    skewness = skew(values)
    kurt = kurtosis(values)
    data_range = np.max(values) - np.min(values)
    perc_90 = np.percentile(values, 90)
    perc_10 = np.percentile(values, 10)
    return mean, sd, med, iqr, skewness, kurt, data_range, perc_90, perc_10

# Function to round numbers for readability
# Can handle cases where number is too small and require scientific notation
def round_and_format(value, decimals=4, threshold=1e-4):
    if abs(value) < threshold:
        return f"{value:.2e}"
    else:
        return round(value, decimals)

# Functions to compute bin size for histograms and plot graphs
def compute_bin_size(numbers):
    bins = int(np.ceil(np.log2(len(numbers)) + 1))
    return bins

def create_histogram_and_display_metrics(numbers1, numbers2, label1, label2, metrics1, metrics2):
    mean1, sd1, med1, iqr1, skew1, kurt1, range1, perc_90_1, perc_10_1 = metrics1
    mean2, sd2, med2, iqr2, skew2, kurt2, range2, perc_90_2, perc_10_2 = metrics2

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot histograms
    axs[0].hist(numbers1, bins=compute_bin_size(numbers1), alpha=0.5, 
                label=f'{label1} - Mean: {mean1:.2f}, SD: {sd1:.2f}, Skew: {skew1:.2f}, Kurtosis: {kurt1:.2f}, Median: {med1:.2f}, IQR: {iqr1:.2f}, Range: {range1:.2f}, P90: {perc_90_1:.2f}, P10: {perc_10_1:.2f}', 
                edgecolor='black')
    axs[0].hist(numbers2, bins=compute_bin_size(numbers2), alpha=0.5, 
                label=f'{label2} - Mean: {mean2:.2f}, SD: {sd2:.2f}, Skew: {skew2:.2f}, Kurtosis: {kurt2:.2f}, Median: {med2:.2f}, IQR: {iqr2:.2f}, Range: {range2:.2f}, P90: {perc_90_2:.2f}, P10: {perc_10_2:.2f}', 
                edgecolor='black')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Frequency')
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1)
    axs[0].grid(True)

    # Plot Q-Q plots
    # These plots can help tell, at a glance, if a given distribution is a Gaussian (or normal) distribution
    qq1 = stats.probplot(numbers1, dist="norm", plot=axs[1])
    line1 = axs[1].get_lines()[1]
    line1.set_color('green')  # Trendline for model_1
    line1.set_linestyle('--')  # Set dashed linestyle
    scatter1 = axs[1].get_lines()[0]
    scatter1.set_color('blue')  # Data points for model_1
    scatter1.set_markersize(scatter1.get_markersize() * 0.5)  # Reduce size of data points
    scatter1.set_label(f'{label1} Data')
    line1.set_label(f'{label1} Trendline')

    qq2 = stats.probplot(numbers2, dist="norm", plot=axs[1])
    line2 = axs[1].get_lines()[3]
    line2.set_color('red')  # Trendline for model_2
    line2.set_linestyle('--')  # Set dashed linestyle
    scatter2 = axs[1].get_lines()[2]
    scatter2.set_color('orange')  # Data points for model_2
    scatter2.set_markersize(scatter2.get_markersize() * 0.5)  # Reduce size of data points
    scatter2.set_label(f'{label2} Data')
    line2.set_label(f'{label2} Trendline')

    # Add special markers at the start and end points of the trendlines without adding them to the legend
    # Done so the trendline can be seen, at a glance, whenever the datapoints obscure it
    common_markersize = 20
    start_point1 = [line1.get_xdata()[0], line1.get_ydata()[0]]
    end_point1 = [line1.get_xdata()[-1], line1.get_ydata()[-1]]
    axs[1].plot(start_point1[0], start_point1[1], 'gx', markersize=common_markersize)  # Green x for start point
    axs[1].plot(end_point1[0], end_point1[1], 'gx', markersize=common_markersize)  # Green x for end point

    start_point2 = [line2.get_xdata()[0], line2.get_ydata()[0]]
    end_point2 = [line2.get_xdata()[-1], line2.get_ydata()[-1]]
    axs[1].plot(start_point2[0], start_point2[1], 'r+', markersize=common_markersize)  # Red + for start point
    axs[1].plot(end_point2[0], end_point2[1], 'r+', markersize=common_markersize)  # Red + for end point

    # Remove the default title
    # Otherwise, a ghost title sits behind the legend, haunts your plot, and looks bad!
    axs[1].set_title('')

    # Calculate and display the slopes of the trendlines
    slope1 = (end_point1[1] - start_point1[1]) / (end_point1[0] - start_point1[0])
    slope2 = (end_point2[1] - start_point2[1]) / (end_point2[0] - start_point2[0])
    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.7)
    axs[1].text(0.95, 0.1, f'{label1} Slope: {slope1:.3f}', horizontalalignment='right', verticalalignment='center', transform=axs[1].transAxes, color='green', bbox=bbox_props)
    axs[1].text(0.95, 0.05, f'{label2} Slope: {slope2:.3f}', horizontalalignment='right', verticalalignment='center', transform=axs[1].transAxes, color='red', bbox=bbox_props)

    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    axs[1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for legend
    plt.show()

# Function to calculate PCA-based weights
def calculate_pca_weights(metrics, num_components=None):
    scaler = StandardScaler()
    standardized_metrics = scaler.fit_transform(metrics)
    pca = PCA()
    pca.fit(standardized_metrics)
    explained_variance_ratios = pca.explained_variance_ratio_
    
    # Use cumulative explained variance if num_components is specified
    if num_components:
        cumulative_variance = np.cumsum(explained_variance_ratios)
        weights = cumulative_variance / np.sum(cumulative_variance)
    else:
        weights = explained_variance_ratios
    
    # Ensure weights match the number of metrics
    if len(weights) < metrics.shape[1]:
        weights = np.pad(weights, (0, metrics.shape[1] - len(weights)), 'constant')
    weights = weights / np.sum(weights)  # Normalize weights
    return weights

# Function to calculate inverse variance-based weights
def calculate_inverse_variance_weights(metrics):
    variances = np.var(metrics, axis=0)
    inverse_variances = 1 / variances
    weights = inverse_variances / np.sum(inverse_variances)
    return weights

# Function to calculate AHP-based weights
# Order of metric ranks was guided by the choices and subsequent importance ordering 
# implied in the Subjective Weights down below
def calculate_ahp_weights():
    # Comparison matrix for 9 metrics
    comparison_matrix = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1/2, 1, 2, 3, 4, 5, 6, 7, 8],
        [1/3, 1/2, 1, 2, 3, 4, 5, 6, 7],
        [1/4, 1/3, 1/2, 1, 2, 3, 4, 5, 6],
        [1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5],
        [1/6, 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4],
        [1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1, 2, 3],
        [1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1, 2],
        [1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1]
    ])

    # Normalize the comparison matrix
    column_sums = np.sum(comparison_matrix, axis=0)
    normalized_matrix = comparison_matrix / column_sums

    # Calculate the weights
    weights = np.mean(normalized_matrix, axis=1)
    return weights

# These "synthetic" normal distributions are generated in order to compare the data distributions with
# Based on the Central Limit Theorem, I expected the data distributions to become more normal as more 
# random image data was added to the histograms. These distributions help determine if this is true, 
# by using the calculated Mean and Standard Deviation (SD), from the data, for the sythetic normals.
def generate_synthetic_normal_metrics(mean, sd, size):
    synthetic_normal = np.random.normal(mean, sd, size)
    return calculate_metrics(synthetic_normal)

# Simple way to qualitatively see if the distribution resembles its normal counterpart.
# Metric values are compared, 1-to-1, between the data and sythetic sets to see if the data
# is normal or not.
def calculate_fractional_percent_difference(data_metrics, synthetic_metrics):
    return [abs((d - s) / s) * 100 for d, s in zip(data_metrics, synthetic_metrics)]

# Function needed to insert newlines into the logger functions for the terminal and TXT output
def insert_newlines(text, line_length):
    lines = []
    while len(text) > line_length:
        split_pos = text[:line_length].rfind(' ')
        if split_pos == -1:
            split_pos = line_length
        lines.append(text[:split_pos])
        text = text[split_pos:].strip()
    lines.append(text)
    return '\n'.join(lines)
# ===== FUNCTION ZOO =======================================

###########################MAIN#############################
def main():
    # Define metric names
    metric_names = ["Mean", "SD", "Median", "IQR", "Skewness", "Excess Kurtosis", "Range", "P90", "P10"]

    # Read data from dataset TXT files
    try:
        values_1 = read_data_from_directory(dir_model_1)
        values_2 = read_data_from_directory(dir_model_2)
    except Exception as e:
        logger.info(f"Error reading files: {e}")
        return

    if values_1.size == 0 or values_2.size == 0:
        logger.info("Error: One or both of the data sets are empty due to reading issues.")
        return

    # Calculate metrics for both models
    metrics_1 = calculate_metrics(values_1)
    metrics_2 = calculate_metrics(values_2)
    
    # Generate synthetic normal distributions, based on dataset Means and SDs, and
    # then calculate the resulting sythetic metrics for comparison
    synthetic_metrics_1 = generate_synthetic_normal_metrics(metrics_1[0], metrics_1[1], len(values_1))
    synthetic_metrics_2 = generate_synthetic_normal_metrics(metrics_2[0], metrics_2[1], len(values_2))

    # Calculate fractional percent differences
    fractional_percent_diff_1 = calculate_fractional_percent_difference(metrics_1, synthetic_metrics_1)
    fractional_percent_diff_2 = calculate_fractional_percent_difference(metrics_2, synthetic_metrics_2)
    
    # Calculate Shapiro-Wilk test for determining normality of a given distribution
    # This is a more rigourous test of normality besides the percent differences
    shapiro_test_1 = shapiro(values_1)
    shapiro_test_2 = shapiro(values_2)

    # Combine metrics for weight calculation
    combined_metrics = np.array([metrics_1, metrics_2])

    # Uniform Weights
    num_metrics = combined_metrics.shape[1]
    uniform_weights = np.ones(num_metrics) / num_metrics

    # Subjective Weights
    # These are based on my naive personal experience and preference of the importance
    # of each of the metrics. Feel free to adjust this based on your experience.
    subjective_weights = np.array([0.25, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05, 0.025, 0.025])

    # PCA-based Weights (considering cumulative variance for all components)
    pca_weights = calculate_pca_weights(combined_metrics, num_components=combined_metrics.shape[1])

    # Inverse Variance-Based Weights
    inverse_variance_weights = calculate_inverse_variance_weights(combined_metrics)

    # AHP-Based Weights
    ahp_weights = calculate_ahp_weights()

    # Calculate General Accuracy Scores
    # I was not sure which method was producing the most reliable output
    # To combat this, I'm simply going to brute force the issue with various averages
    # taking a 'final' average of the averages to come to a final conclusion
    def calculate_scores(weights):
        general_accuracy_score_1 = np.dot(metrics_1, weights)
        general_accuracy_score_2 = np.dot(metrics_2, weights)
        return general_accuracy_score_1, general_accuracy_score_2

    uniform_score_1, uniform_score_2 = calculate_scores(uniform_weights)
    subjective_score_1, subjective_score_2 = calculate_scores(subjective_weights)
    pca_score_1, pca_score_2 = calculate_scores(pca_weights)
    inverse_variance_score_1, inverse_variance_score_2 = calculate_scores(inverse_variance_weights)
    ahp_score_1, ahp_score_2 = calculate_scores(ahp_weights)

    # Determine which model wins for each method
    wins_1 = 0
    wins_2 = 0

    if uniform_score_1 < uniform_score_2:
        wins_1 += 1
    else:
        wins_2 += 1

    if subjective_score_1 < subjective_score_2:
        wins_1 += 1
    else:
        wins_2 += 1

    if pca_score_1 < pca_score_2:
        wins_1 += 1
    else:
        wins_2 += 1

    if inverse_variance_score_1 < inverse_variance_score_2:
        wins_1 += 1
    else:
        wins_2 += 1

    if ahp_score_1 < ahp_score_2:
        wins_1 += 1
    else:
        wins_2 += 1

    # Average General Accuracy Scores
    average_score_1 = np.mean([uniform_score_1, subjective_score_1, pca_score_1, inverse_variance_score_1, ahp_score_1])
    average_score_2 = np.mean([uniform_score_2, subjective_score_2, pca_score_2, inverse_variance_score_2, ahp_score_2])

    # Log the weights and scores for verification
    logger.info("Uniform Weights: %s", [round_and_format(w) for w in uniform_weights])
    logger.info("Subjective Weights: %s", [round_and_format(w) for w in subjective_weights])
    logger.info("PCA-Based Weights: %s", [round_and_format(w) for w in pca_weights])
    logger.info("Inverse Variance-Based Weights: %s", [round_and_format(w) for w in inverse_variance_weights])
    logger.info("AHP-Based Weights: %s", [round_and_format(w) for w in ahp_weights])

    logger.info("\n" + "="*50)
    logger.info("Metric Values")
    logger.info("="*50)

    metric_dict_1 = {name: round_and_format(value) for name, value in zip(metric_names, metrics_1)}
    metric_dict_2 = {name: round_and_format(value) for name, value in zip(metric_names, metrics_2)}

    logger.info("Metrics for model_1: %s", metric_dict_1)
    logger.info("Metrics for model_2: %s", metric_dict_2)

    synthetic_metric_dict_1 = {name: round_and_format(value) for name, value in zip(metric_names, synthetic_metrics_1)}
    synthetic_metric_dict_2 = {name: round_and_format(value) for name, value in zip(metric_names, synthetic_metrics_2)}

    logger.info("\nSynthetic Metrics for model_1: %s", synthetic_metric_dict_1)
    logger.info("Synthetic Metrics for model_2: %s", synthetic_metric_dict_2)

    logger.info("\nSynthetic Normal Distribution Metrics Comparison for model_1 (percent differences):")
    for name, diff in zip(metric_names, fractional_percent_diff_1):
        logger.info(f"{name}: {round_and_format(diff):.2f}%")

    logger.info(f"\nShapiro-Wilk Test for model_1: W={shapiro_test_1.statistic:.5f}, p-value={shapiro_test_1.pvalue:.5f}")
    if shapiro_test_1.pvalue > 0.05 and shapiro_test_1.statistic > 0.95:
        logger.info("Normal Status: EXPECTED!")
    elif shapiro_test_1.pvalue <= 0.05 and shapiro_test_1.statistic < 0.95:
        logger.info("Normal Status: NOT expected!")
    else:
        logger.info("Normal Status: ???")

    logger.info("\nSynthetic Normal Distribution Metrics Comparison for model_2 (percent differences):")
    for name, diff in zip(metric_names, fractional_percent_diff_2):
        logger.info(f"{name}: {round_and_format(diff):.2f}%")

    logger.info(f"\nShapiro-Wilk Test for model_2: W={shapiro_test_2.statistic:.5f}, p-value={shapiro_test_2.pvalue:.5f}")
    if shapiro_test_2.pvalue > 0.05 and shapiro_test_2.statistic > 0.95:
        logger.info("Normal Status: EXPECTED!")
    elif shapiro_test_2.pvalue <= 0.05 and shapiro_test_2.statistic < 0.95:
        logger.info("Normal Status: NOT expected!")
    else:
        logger.info("Normal Status: ???")

    logger_common_symbol_length = 56
    new_line_char_lim = 56
    message1 = "Model_1 has the best possible accuracy score based on the above methods and statistics!"
    message2 = "Model_2 has the best possible accuracy score based on the above methods and statistics!"
    message3 = "Model determination is mixed, please check the 'Better Model' for more details."
    message4 = "No model has a definitive best possible accuracy score based on the above methods and statistics."
    formatted_message1 = insert_newlines(message1, new_line_char_lim)
    formatted_message2 = insert_newlines(message2, new_line_char_lim)
    formatted_message3 = insert_newlines(message3, new_line_char_lim)
    formatted_message4 = insert_newlines(message4, new_line_char_lim)

    logger.info("\n" + "="*logger_common_symbol_length)
    logger.info("General Accuracy Scores - Uniform Weights")
    logger.info("="*logger_common_symbol_length)
    logger.info("General Accuracy Score for model_1: %.4f", round_and_format(uniform_score_1))
    logger.info("General Accuracy Score for model_2: %.4f", round_and_format(uniform_score_2))

    logger.info("\n" + "="*logger_common_symbol_length)
    logger.info("General Accuracy Scores - Subjective Weights")
    logger.info("="*logger_common_symbol_length)
    logger.info("General Accuracy Score for model_1: %.4f", round_and_format(subjective_score_1))
    logger.info("General Accuracy Score for model_2: %.4f", round_and_format(subjective_score_2))

    logger.info("\n" + "="*logger_common_symbol_length)
    logger.info("General Accuracy Scores - PCA-Based Weights")
    logger.info("="*logger_common_symbol_length)
    logger.info("General Accuracy Score for model_1: %.4f", round_and_format(pca_score_1))
    logger.info("General Accuracy Score for model_2: %.4f", round_and_format(pca_score_2))

    logger.info("\n" + "="*logger_common_symbol_length)
    logger.info("General Accuracy Scores - Inverse Variance-Based Weights")
    logger.info("="*logger_common_symbol_length)
    logger.info("General Accuracy Score for model_1: %.4f", round_and_format(inverse_variance_score_1))
    logger.info("General Accuracy Score for model_2: %.4f", round_and_format(inverse_variance_score_2))

    logger.info("\n" + "="*logger_common_symbol_length)
    logger.info("General Accuracy Scores - AHP-Based Weights")
    logger.info("="*logger_common_symbol_length)
    logger.info("General Accuracy Score for model_1: %.4f", round_and_format(ahp_score_1))
    logger.info("General Accuracy Score for model_2: %.4f", round_and_format(ahp_score_2))

    logger.info("\n" + "="*logger_common_symbol_length)
    logger.info("Averaged General Accuracy Scores")
    logger.info("="*logger_common_symbol_length)
    logger.info("Average General Accuracy Score for model_1: %.4f", round_and_format(average_score_1))
    logger.info("Average General Accuracy Score for model_2: %.4f", round_and_format(average_score_2))

    logger.info("\n" + "="*logger_common_symbol_length)
    logger.info("Better Model")
    logger.info("="*logger_common_symbol_length)
    if average_score_1 < average_score_2:
        logger.info("Model_1 has a better average general accuracy score.")
    else:
        logger.info("Model_2 has a better average general accuracy score.")

    logger.info(f"Model_1 won {wins_1}/5 methods")
    logger.info(f"Model_2 won {wins_2}/5 methods")

    # Determine the final outcome
    logger.info("\n" + "-"*logger_common_symbol_length)
    logger.info("Final Outcome")
    logger.info("-"*logger_common_symbol_length)
    if average_score_1 < average_score_2 and wins_1 > wins_2:
        logger.info(formatted_message1)
    elif average_score_2 < average_score_1 and wins_2 > wins_1:
        logger.info(formatted_message2)
    elif (average_score_1 < average_score_2 and wins_2 > wins_1) or (average_score_2 < average_score_1 and wins_1 > wins_2):
        logger.info(formatted_message3)
    else:
        logger.info(formatted_message4)

    # Save output to file
    with open("output_stats.txt", "w") as f:
        f.write(log_stream.getvalue())

    # Plot histograms and display metrics
    create_histogram_and_display_metrics(values_1, values_2, 'Model_1', 'Model_2', metrics_1, metrics_2)
###########################MAIN#############################

# Section needed if I ever want to incorperate parallel processing
if __name__ == "__main__":
    main()
    