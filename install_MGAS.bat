@echo off
REM Check if Python is installed
python --version
IF ERRORLEVEL 1 (
    echo Python is not installed. Please install Python Version 3.7, or later.
    pause
    exit /B
)

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

REM Check if shape_predictor_5_face_landmarks.dat exists
if exist "%DLIB_DIR%\shape_predictor_5_face_landmarks.dat" (
    echo shape_predictor_5_face_landmarks.dat already exists. Skipping download.
) else (
    set DOWNLOAD_NEEDED=1
)

REM Check if dlib_face_recognition_resnet_model_v1.dat exists
if exist "%DLIB_DIR%\dlib_face_recognition_resnet_model_v1.dat" (
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
) else (
    echo All files are already present. No downloads needed.
)

REM Separator and final message
echo =====================================================================
echo.
echo All files downloaded.
echo.
echo Setup completed successfully.
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
