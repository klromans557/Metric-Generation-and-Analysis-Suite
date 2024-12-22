@echo off
REM Check if Python is installed
python --version
IF ERRORLEVEL 1 (
    echo Python is not installed. Please install Python Version 3.7, or later.
    pause
    exit /B
)

REM Create necessary directory structure if it doesn't exist
echo Checking and creating required directory structure...
if not exist CACHE mkdir CACHE
if not exist DIR mkdir DIR
if not exist DIR\images mkdir DIR\images
if not exist DIR\output mkdir DIR\output
if not exist DIR\references mkdir DIR\references
if not exist DLIB mkdir DLIB
if not exist LOGS mkdir LOGS
IF ERRORLEVEL 1 (
    echo Error: Failed to create one or more directories.
    pause
    exit /B
)

echo Directory structure checked/created successfully.

REM Create a virtual environment
python -m venv venv
IF ERRORLEVEL 1 (
    echo Failed to create virtual environment. Please check your Python installation.
    pause
    exit /B
)

REM Activate the virtual environment
call .\venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install required dependencies
pip install -r requirements.txt
IF ERRORLEVEL 1 (
    echo Failed to install required dependencies. Please check the requirements.txt file and your internet connection.
    pause
    exit /B
)

REM Check if shape_predictor_68_face_landmarks.dat exists
set DLIB_DIR=DLIB
set SHAPE_FILE=%DLIB_DIR%\shape_predictor_68_face_landmarks.dat
set RESNET_FILE=%DLIB_DIR%\dlib_face_recognition_resnet_model_v1.dat

if exist "%SHAPE_FILE%" (
    echo shape_predictor_68_face_landmarks.dat already exists. Skipping download.
) else (
    set DOWNLOAD_NEEDED=1
)

if exist "%RESNET_FILE%" (
    echo dlib_face_recognition_resnet_model_v1.dat already exists. Skipping download.
) else (
    set DOWNLOAD_NEEDED=1
)

REM Run the combined Python script if any file is missing
if defined DOWNLOAD_NEEDED (
    echo Downloading files...
    python HF_model_download.py
    if %errorlevel% neq 0 (
        echo Error: Failed to download one or more files.
        pause
        exit /b %errorlevel%
    )
)

REM Verify that the necessary .dat files are present
if exist "%SHAPE_FILE%" (
    if exist "%RESNET_FILE%" (
        set SETUP_SUCCESS=1
    )
)

REM Conditional success message
if defined SETUP_SUCCESS (
    echo =====================================================================
    echo.
    echo All files downloaded successfully, and setup completed.
    echo You may now use the app with the run_GUI BAT!
    echo =====================================================================
    echo.
    echo Make sure you have your model image folders in the 'images' directory
    echo and your fixed set of reference images in the 'references' directory
    echo in order to use the CREATE face-distance script
    echo. 
    echo Put some, or all, of the example data directories in 'output', OR
    echo first run the CREATE script with your images, to test OR use 
    echo the BULK analysis script
    echo =====================================================================
    pause
) else (
    echo =====================================================================
    echo.
    echo Error: One or more required .dat files are missing.
    echo Please check your setup and try running the script again.
    echo =====================================================================
    pause
    exit /B
)

pause
