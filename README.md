# Metric Generation and Analysis Suite

A set of Python scripts that I developed to help me train AI image models using large datasets of facial embedding data, determining a generalized accuracy score based on various statistical methods. Given at least two models represented by folders containing their generated images, these scripts create a set of facial similarity metric data as simple Python lists for each image. With this data a carefully selected set of statistical metrics and weighting schemes are used to generate the score, and a couple of plots are created to quickly visualize the results. 

The intention is to use the scripts to compare two Generative AI image models (cf. Stable Diffusion) by analyzing large sets of randomly generated images against a finite set of reference images of the trained subject. Such a comparison can be crucial in determining the effect of a single change in training methodology and its overall effect on model output. For example, seeing the effect of changing the Loss Weight function between 'Constant' and 'Min_SNR_Gamma' on what is otherwise the same dataset and hyperparameters. Ultimately, the success of a model is determined by its ability to reliably reproduce the likeness of the trained subject, and this script is my first attempt to quantitatively address issues related to that.

¡NOTE! The repo now contains a GUI and script that can perform the similarity metric analysis without the need for any external apps/nodes/extensions; it is now self-contained!

Built with the help of GPT-4o; thanks Zedd! (>'.')>[<3] 

Feel free to use, share, and modify this suite to suit your needs.
Made "for fun", and shared completely for free to those who love GenAI.
<(˶ᵔᵕᵔ˶)>

Please read the 'PLEASE_READ_ME' TXTs in the main repo and example directories for more information, details, and results from three of my experiments using this suite of scripts. Also check out the associated Civitai article, [here](https://civitai.com/articles/6327/statistical-analysis-of-ai-models-an-overview-of-the-mgas-comparison-method-and-scripts)

![screenshot](EXAMPLES/Example_Figures/Example_Figure_0_v2-3GUI.png)

## Table of Contents

- [Changelog](#changelog)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Changelog

### [2.5] - 2024/12/19
This update focuses on properly implementing the facial recognition workflow from the methodology section and optimizing the parallel processing to increase speed.

#### Added:
- Implemented the Alignment and Normalization steps through the `align_face` function along with DLib's `get_face_chip`. Faces are now properly preprocessed before the embeddings are calculated.
- Included the 68-point landmark model for use in the Alignment step and increased accuracy. This now replaces the previously used 5-point model in all use cases.

#### Fixed:
- Fixed the absolute mess that the parallel processing functionalities in the `process_images` function loop was. This has significantly increased processing speed and improved CPU usage.
- Fixed the BULK script to use alphanumerical sorting rather than lexicographical.
- Improved logging and error reporting relevant to the `process_log.txt` file.
  
### [2.4] - 2024/12/09
After taking a break I came back and found some issues while testing a new model under relatively fresh install conditions.

#### Added:
- Included the `HF_model_download.py` to handle the download of the two face-recognition DAT files from HuggingFace. This now replaces the janky curl method in the `install_MGAS.bat` file.
- Added huggingface_hub to the `requirements.txt`.

#### Fixed:
- Fixed how the scripts were calling the local venv. Although the venv is activated when the GUI starts, the `gui.py` script itself was not enforcing this when calling the other scripts. Now, the `gui.py` script properly ensures that the venv python is used in the `run_script` method.

### [2.3] - 2024/07/31

#### Changed:
- Changed how model names were assigned. Now, model names follow their folder names instead of the generic 'model_x' format, and this change should be reflected in the corresponding LOGS.
- Changed how model names were sorted with regards to the 'models to compare directly' variable. Now, model order is prepended onto folder names such that which two models are chosen is no longer ambiguous, e.g. 'model_name' --> '1_model_name'. Python was sorting by name in a slightly different way from Windows, and this was resulting in the incorrect folders being assigned to the 'models to compare directly' variable; i.e. script order was not matching what the user could see in the 'output' folder itself.

### [2.2] - 2024/07/30

#### Changed:
- Changed how the results of the Round-Robin were determined and reported. The convoluted MGAS system has been replaced with a simple, but effective, tally system.
- Results of the Round-Robin now "make sense" and should now accurately reflect the results from Two-Model Direct Comparisons; previous method was far too unreliable and unpredictable.
- Changed Round-Robin to only report the wins of the top three models.
- Changed the GAS back to its intended definition, that of a score determined by an individual weight method.
- Changed the MGAS back to its intended definition, that of a simple Mean-GAS for each model based on all weight methods. 

### [2.1] - 2024/07/29

#### Added:
- Included the `optimize_facedata_weights.py` used to generated associated weights in `_EXTRAS` for user reference
- Included additional experimental data in `Example_Data_Directories` over extracted LoRA Rank(Dim) and Alpha combinations,
  as well as updated example figures and logs
- Added `Weighted Rank Sum`-Based Weights to list of used methods
- Added my personal `Meat-N-Potatoes`-Based Weights as a "sanity check"

### Changed:
- Changed the previous 'Subjective' Weights to the `Optimized` Weights based on included optimization scheme and new order of import of the metrics
- Changed the PCA method to the `Robust Principal Component Analysis`-Based Weights to better handle noisy data
- Changed the Kurtosis and Skewness to, `Sc_Kurtosis` & `Sc_Skewness`, which are scaled versions that have the correct "units" as the data

#### Fixed:
- ¡HUGE! Metric redundancies were curtailed through correlation analysis, and a minimum set of metrics was chosen through optimization
- ¡HUGE! All metrics have been scaled and standardized to ensure that they have the same "units" as each other and the data
- ¡HUGE! GAS/MGAS calculations fixed to better capture the notion/maxim that, 'more overall low-valued data, the better the distribution'
- Fixed some GUI options not giving the appropriate warning when the user puts in inappropriate values; 'Number of Processes' & 'Models to Compare'
- Fixed hyphen in GUI start message that was "too dang close"

### [2.0] - 2024/07/26

#### Added:
- New `create_facedistance_data.py` script to carry out data generation without the need for an external app.
- New `gui.py` script for GUI interface; use `run_GUI.bat` to begin app and use both main scripts.
- Added `Multi-Model Round-Robin` tournament style comparison capability which allows for more than two models to be compared at once
  (old way is now called Two-Model Direct Comparison).
- Added the `Kolmogorov-Smirnov` normality test; code will switch from the `Shapiro-Wilk` test when data sets get sufficiently large (~5000)
- Added `install_MGAS.bat` to create venv with required Python dependencies and to automatically download DLib models needed for the CREATE script.
- Added additional information via two `PLEASE_READ_ME.txt` files, one in main and other in example directories.
  
### Changed:
- Reordered the `order_of_import_of_metrics` variable in the BULK script.
- Changed what information is made immediately available, via terminal/GUI; check `LOGS` directory for detailed logs.
- Directory path assignment, and important user defined variables, now accessed through GUI; no longer need to edit any of the scripts!

### [1.0] - 2024/07/23
_First release._

## Installation

1. Please have Python 3.7, or later, installed (I use 3.10.11, as do many other AI apps). You can download it from [python.org](https://www.python.org/downloads/) in the "Looking for a specific release?" section by scrolling down (3.10.11 at release date, April 5, 2023).

2. Clone the repository and go into the created folder:
    ```sh
    git clone https://github.com/klromans557/Metric-Generation-and-Analysis-Suite
    cd Metric-Generation-and-Analysis-Suite
    ```

3. Install the required dependencies:

    Use the provided `install_MGAS.bat` file to install required Python dependencies and DLib models (~ 100MB).
    
## Usage

To use the script and analyze the face-distance data, follow these steps (see `PLEASE_READ_ME.txt` files for more details):

0. GitHub does not like empty directories, so please delete any text files called `DELETE_ME.txt` in the `DIR` folders.
   
1. Use the provided `run_GUI.bat` file to open the GUI and use the scripts.
   
2. Make sure your image folders and fixed set of reference images are in the correct directories within `DIR` before clicking the `Run Create Script` button.
   This script will generate the face similarity metric data in the `output` folder.
   - Within images folder: place folders of images that will represent "models" to be tested. For example, a folder called "dpmpp" and another called "euler", each filled with 10 images generated using those respective samplers (all other parameters fixed).
   - Within references folder: place a fixed set of reference images to compare the "models" to. Data for each reference is collected together for the corresponding tested model. For example, use 5 images of your main subject here. Each model will then get 10x5=50 data points for the BULK statistics. Results improve with more images/refs!

4. Data created, or placed, into the `output` folder is accessible to the BULK script. Use `Run Bulk Script` button to perform statistics on data.
   - Data on the metrics/weights, results of the comparisons, and the tournament winner will then be displayed in the GUI text box.
   - All created data, including the normality tests not usually shown, are stored in the LOGS directory.

6. Press the `Open Last Saved Figure` button to open the last figure made by the BULK script. Note that any changes will require the BULK script to be re-run
   in order to update the figure.
   - Use the `Models to Compare` GUI variable to change which two models are plotted in the figure; models are ordered by name, ascending,
     i.e. the default '1,2' compares the first and second model folders.
   - Use the `Number of Processes` GUI variable to change how many CPU cores/threads are used for processing images with the CREATE script

## Methodology

### Facial Recognition

The heart of the method relies on the similarity facial embedding distance data created by the face recognition models. Roughly speaking, the model finds 68 landmarks on the reference's face, i.e. eyes, nose tip, mouth corners, etc., and assigns a distance value (e.g. 0.3). The process unfolds as:

1. Face Detection: The model first identifies the general region of the face, just like a bounding box.
2. Landmark Detection: Then the landmarks are identified within the face region.
3. Alignment: The face is then "aligned" using the landmarks. This step tries to account for various poses by adjusting faces to a standard forward-facing form. In other words, a tilted face is adjusted such that the eyes are horizontal and the face upright.
4. Normalization: The aligned face is cropped & resized to fit a model-defined standard size (e.g. 150x150 pixels). This ensures uniform processing.
5. Feature Extraction: The normalized face is then analyzed for unique features. These features then form a kind of "fingerprint" of the face which can be used for recognition.
6. Recognition: Finally, the extracted features are compared to a known database of faces. In the case of MGAS, this database is made up of the reference images placed by the user. The system analyzes how similar the faces are and generates a distance value based on this association (e.g. 0.3). The lower the value the closer the tested image matches a given reference. This is the origin of the maxim, "Lower is better!"

From a given set of images to test and another set of reference images to form the database, a large set of statistical data can be generated by these similarity measurements.

### Metrics

Once the base statistical data is generated it can be analyzed with various metrics (see the main repo image for example of a histogram created from such data). I chose a set of six metrics, i. Median, ii. 90th-Percentile (P90), iii. Scaled Skewness, iv. Scaled Kurtosis, v. Interquartile range (IQR), and the Standard Deviation (SD), with the order set by importance. This set was chosen to best minimize redundant correlations between the metrics and to maximize accounting for distribution shape and position. A "good" distribution is not only centered on a low value (i.e. low median), but it also has as more data below the median than above it.

1. Median: Is the middle value of the dataset when the set is ordered. This behaves effectively the same as the Mean (or Average) for the kind of data we will analyze with MGAS, i.e. represents the center of the histogram, but it is less susceptible to extreme values and outliers. For example, when viewing the main repo histograms, the median is around 0.33 which corresponds to the point on the x-axis under the highest part of the distribution.
2. 90th-Percentile (P90): Is the value below which 90% of the data points fall, or in other words, the highest 10% of the data is beyond this point. This value helps describe the tail of the distribution and gives an idea of the magnitude of the higher-end values.
3. Scaled Skewness: The regular Skewness measures the symmetry of the distribution. For a perfectly symmetric distribution, like the Gaussian, this metric is 0. Positive Skewness values correspond to a longer right-side tail, while negative values are for a longer left-side tail. I have scaled these values to ensure they have the same "units" as the other metrics to keep comparisons meaningful.
4. Scaled Kurtosis: The regular Kurtosis (or sometimes Excess Kurtosis) measures the "tailedness" of a distribution or how extreme the outliers are. A Gaussian distribution has a Kurtosis of 3 (Excess Kurtosis of 0), meaning it has a moderate tail length. I have scaled these values to ensure they have the same "units" as the other metrics to keep comparisons meaningful.
5. Interquartile Range (IQR): Is the range of values between the 25-th and 75-th percentiles. It focuses on the spread of the middle 50% of the data, ignoring the tails, and gives a sense of distribution variability and data compactness.
6. Standard Deviation (SD): Is the measure of the average spread of data points around the mean. It also gives a sense of distribution variability taking into account the tails.

### Weights

With the metrics at hand, we can now use them in various statistical weighting schemes, i.e. different types of averages. Using more than one method leverages the advantages/disadvantages of different distributions without shoe-horning a "one-size-fits-all" solution. Each method will generate an individual GAS score.

1. Uniform Weights: Each metric is treated equally with the same weight value (1/6 ~ 0.167) assigned to all. This is the simplest approach as it is just a basic average.
2. Optimized Weights: These weights were calculated based on the order of import of the metrics and to reduce redundancy from metrics that contain similar information. The script used to optimize is located in the _EXTRAS directory.
3. Weighted Rank Sum Weights: Metrics are ranked based on their values for each dataset, and then the ranks are weighted. This scheme emphasizes the relative position of metrics rather than raw values, making it less sensitive to extreme values and outliers.
4. Inverse Variance Sum: Metrics with lower variability (variance) are given higher weights since they are considered more reliable. This gives stable metrics a larger impact on the analysis.
5. Analytic Hierarchy Process weights: A structured method where metrics are compared pairwise to determine their relative importance.
6. Robust Principal Component Weights: This statistical technique focuses on combinations of metrics that cover the most variation in the data. The weights are derived from these combinations, focusing on capturing the key features of the dataset while reducing noise.
7. "Meat-N-Potatoes" Weights: My sanity check weights which only use the Median and P90 weighted equally (1/2 = 0.5).

After all the weights are calculated, the final MGAS score is found by a simple average of the seven weight scheme values.

### Direct Comparisons and Round-Robin Tournament

1. Two-Model Direct Comparison: The two models from the `Models to Compare` GUI variable will be compared, and their respective GAS and MGAS scores calculated. The model which won the most weight scheme comparisons and has the lowest MGAS value is declared the winner.
2. Multi-Model Round-Robin Tournament: All models used by MGAS will be compared here by pairwise direct comparison competitions. The model that wins the most comparisons is declared the winner, with the top three listed in the GUI as well.

## Contributing

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, please follow these steps:

1. Fork the repository.
2. Create a new branch with a descriptive name (`git checkout -b my-feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin my-feature-branch`).
5. Open a pull request.

Please ensure your code adheres to the existing style and includes appropriate tests.

### Reporting Issues

If you find a bug or have a feature request, please create an issue [here](https://github.com/klromans557/Metric-Generation-and-Analysis-Suite/issues).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Matteo Spinelli](https://github.com/cubiq/ComfyUI_FaceAnalysis) for creating the ComfyUI custom node that I used as inspiration for the CREATE script
  and for hosting the DLib model files on HuggingFace.
- [OpenAI](https://www.openai.com) & [Alibaba Cloud](https://www.alibabacloud.com) for providing guidance and assistance in developing this project.
- [GitHub](https://github.com) for hosting the repository.
- [Dr. Furkan Gözükara](https://civitai.com/user/SECourses) for sharing his scripts through the SECourses Civitai, Patreon, associated Discord server and YouTube channel.
  These resources were invaluable to me during the development of this project and served as guides/templates for creating such scripts.
