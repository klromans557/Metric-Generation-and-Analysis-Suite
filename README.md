# Metric Generation and Analysis Suite

Please read the 'PLEASE_READ_ME' TXTs in the main repo and example directories for more information, details, and results from three of my experiments using this suite of scripts. Also check out the associated Civitai article, [here](https://civitai.com/articles/6327/statistical-analysis-of-ai-models-an-overview-of-the-mgas-comparison-method-and-scripts)

A Python script I designed to help me analyze large datasets of face-embedding distances, determining a generalized accuracy score based on various statistical methods. This script creates for (at least) two folders a set of facial similarity metric data, contained as lists of values (e.g. \[1,2,3,...\]) for each image, 
and calculates multiple statistical metrics and visualizes the results. 

The intention is to use the scripts to compare two Stable Diffusion models by analyzing the L2_Norm, Euclidean, and/or Cosine facial-embed distances (one-type at a time, L2_Norm used in example data) 
of large sets of randomly generated images against a finite set of reference images of the trained subject. 
Such a comparison can be crucial in determining the effect of a single change in training methodology and its overall effect on model output; for example, seeing the effect of changing the Loss Weight function between 'Constant' and 'Min_SNR_Gamma' on what is otherwise the same dataset and hyperparameters. 
Ultimately, the success of a model is determined by its ability to reliably reproduce the likeness of the trained subject, and this script is my first attempt to quantitatively address issues related to that.

¡NOTE! The repo now contains a GUI and script that can perform the similarity metric analysis without the need for any external apps/nodes/extensions; it is now self-contained!

Built with the help of GPT-4o; thanks Zedd! (>'.')>[<3] 

Feel free to use, share, and modify this suite to suit your needs.
Made "for fun", and shared completely for free to those who love GenAI.
<(˶ᵔᵕᵔ˶)>

![screenshot](EXAMPLES/Example_Figures/Example_Figure_0_v2GUI.png)

## Table of Contents

- [Changelog](#changelog)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Changelog

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

1. Please have Python 3.7, or later, installed. You can download it from [python.org](https://www.python.org/downloads/).

2. Clone the repository and go into the created folder:
    ```sh
    git clone https://github.com/klromans557/Metric-Generation-and-Analysis-Suite
    cd Metric-Generation-and-Analysis-Suite
    ```

3. Install the required dependencies:

    Use the provided `install_MGAS.bat` file to install required Python/DLib dependencies/models.
    
## Usage

To use the script and analyze the face-distance data, follow these steps (see `PLEASE_READ_ME.txt` files for more details):

0. GitHub does not like empty directories, so please delete any text files called `DELETE_ME.txt` in the `DIR` folders.
   
1. After installing required dependencies, use the provided `run_GUI.bat` file to open the GUI and use the scripts.
   
2. Make sure your image folders and fixed set of reference images are in the correct directories within `DIR` before clicking the `Run Create Script` button.
   This script will generate the face similarity metric data in the `output` folder.

3. Data collected, or placed, into the `output` folder is accessible to the BULK script. Use `Run Bulk Script` button to perform statistics on data.

4. Press the `Open Last Saved Figure` button to open the last figure made by the BULK script. Note that any changes will require the BULK script to be re-run
   in order to update the figure.
   - Use the `Models to Compare` GUI variable to change which two models are plotted in the figure; models are ordered by name, ascending,
     i.e. the default '1,2' compares the first and second model folders.
   - Use the `Number of Processes` GUI variable to change how many CPU cores/threads are used for processing images with the CREATE script

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
- [OpenAI](https://www.openai.com) for providing guidance and assistance in developing this project.
- [GitHub](https://github.com) for hosting the repository.
- [Dr. Furkan Gözükara](https://civitai.com/user/SECourses) for sharing his scripts through the SECourses Civitai, Patreon, associated Discord server and YouTube channel.
  These resources were invaluable to me during the development of this project and served as guides/templates for creating such scripts.
