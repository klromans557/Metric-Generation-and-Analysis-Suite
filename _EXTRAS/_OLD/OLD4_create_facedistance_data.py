import os
import logging
import sys
import numpy as np
from PIL import Image
import dlib
import cv2
from scipy.spatial.distance import cosine, euclidean
import numpy.linalg as la
import multiprocessing
from multiprocessing import Pool

############## CREATE FACE-DISTANCE DATA ###################
# A script developed by klromans557 (a.k.a reaper557) with
# the help of GPT-4o. Thanks Zedd! (>'.')>[<3] [7/25/2024]
#
# Shared completely FOR FREE! (MIT - license)
# To those who love GenAI.
# 
# The purpose of this script is to generate the facedistance
# embedding data needed for the statistics script. Place the
# folders with images and a fixed set of reference images 
# into the 'images' and 'references' subdirectories of the 
# 'DIR' folder, respectively. Double-click the appropriate 
# BAT to run it. Face distance data will then be placed into
# the 'output' subdirectory of 'DIR' in 1:1 folders to that 
# of the images.
############################################################

# ===== LOGGING ============================================
# Custom logging function to create information for 'process_log' TXT and terminal window output
# Update the log function to use the LOGS folder
# Define a constant for the logging level
LOG_LEVEL = "INFO"  # Change this to DEBUG, INFO, WARNING, ERROR, or CRITICAL as needed
# Enhanced logging with verbosity levels
def setup_logging():
    log_file = os.path.join(os.getcwd(), 'LOGS', 'process_log.txt')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Log to console as well
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(console_handler)

def log(message, level="INFO"):
    # Use the specified log level (default is INFO)
    getattr(logging, level.lower(), logging.info)(message)

# ===== LOGGING ============================================

# ===== CLASSES ============================================
class DLib:
    def __init__(self, shape_predictor_68_path, face_recognition_model_path):
        self.detector = dlib.get_frontal_face_detector()

        if not os.path.exists(shape_predictor_68_path):
            raise FileNotFoundError(f"68-point model not found: {shape_predictor_68_path}")
        self.shape_predictor_68 = dlib.shape_predictor(shape_predictor_68_path)

        if not os.path.exists(face_recognition_model_path):
            raise FileNotFoundError(f"Face recognition model not found: {face_recognition_model_path}")
        self.face_rec_model = dlib.face_recognition_model_v1(face_recognition_model_path)

    def align_face(self, image, shape):
        try:
            face_chip = dlib.get_face_chip(image, shape, size=150)
            log("Face alignment and preprocessing successful.")
            return face_chip, shape
        except Exception as e:
            log(f"Error during alignment and preprocessing: {e}")
            return None, None

    def get_embedding(self, image):
        try:
            dets = self.detector(image, 1)
            log(f"Detected {len(dets)} faces.")
            if dets:
                shape = self.shape_predictor_68(image, dets[0])
                aligned_image, _ = self.align_face(image, shape)
                if aligned_image is not None:
                    embedding = np.array(self.face_rec_model.compute_face_descriptor(aligned_image))
                    if embedding.size > 0:
                        log("Successfully computed embedding.")
                        return embedding / (np.linalg.norm(embedding) or 1)
            log("Failed to produce valid embedding.")
            return None
        except Exception as e:
            log(f"Error during embedding computation: {e}.")
            return None

# ===== CLASSES ============================================

# ===== GLOBAL INITIALIZER FOR MULTIPROCESSING =============
_global_dlib = None

def init_worker(shape_predictor_68_path, face_recognition_model_path):
    global _global_dlib
    if _global_dlib is None:
        _global_dlib = DLib(
            shape_predictor_68_path,
            face_recognition_model_path
        )

# ===== GLOBAL INITIALIZER FOR MULTIPROCESSING =============

# ===== FUNCTION ZOO =======================================
def get_face_embedding(image_path, shape_predictor_68_path=None, face_recognition_model_path=None):
    global _global_dlib
    if _global_dlib is None and shape_predictor_68_path and face_recognition_model_path:
        _global_dlib = DLib(shape_predictor_68_path, face_recognition_model_path)

    try:
        image = np.array(Image.open(image_path).convert('RGB'))
    except Exception as e:
        log(f"Failed to load image {image_path}: {e}")
        return None

    try:
        embedding = _global_dlib.get_embedding(image)
        return embedding
    except Exception as e:
        log(f"Error processing {image_path}: {e}")
        return None

def compute_distance(embedding1, embedding2, metric):
    if metric == "cosine":
        return cosine(embedding1, embedding2)
    elif metric == "euclidean":
        return euclidean(embedding1, embedding2)
    elif metric == "L2_norm":
        return la.norm(embedding1 - embedding2)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def validate_num_processes(num_processes):
    max_processes = (multiprocessing.cpu_count() // 2) + 2
    if num_processes > max_processes:
        raise ValueError(f"Number of processes ({num_processes}) exceeds the number of recommended ``safe`` CPU cores/threads ({max_processes}).")

def validate_dlib_models(dlib_folder):
    required_files = [
        'shape_predictor_68_face_landmarks.dat',
        'dlib_face_recognition_resnet_model_v1.dat'
    ]

    for file_name in required_files:
        if not os.path.exists(os.path.join(dlib_folder, file_name)):
            raise FileNotFoundError(f"Required DLib file not found: {file_name}")

def process_single_image(args):
    try:
        (image_path, ref_embedding, metric, shape_predictor_68_path, 
         face_recognition_model_path, output_subdir, ref_name) = args

        embedding = get_face_embedding(
            image_path,
            shape_predictor_68_path=shape_predictor_68_path,
            face_recognition_model_path=face_recognition_model_path,
        )
        if embedding is None:
            log(f"Failed to compute embedding for image {image_path}")
            return None

        if ref_embedding is None:
            log(f"Reference embedding is None for image {image_path}")
            return None

        # Compute the distance
        distance = compute_distance(ref_embedding, embedding, metric)
        log(f"Computed distance for {image_path} with {ref_name}: {distance}")

        # Define the output file path
        output_file = os.path.join(output_subdir, f"{ref_name}_distances.txt")

        # Ensure the output file remains a Python list
        try:
            # Load existing data if the file exists
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    existing_data = eval(f.read())  # Safely parse the file content as Python list
            else:
                existing_data = []

            # Append the new result
            existing_data.append(distance)

            # Write the updated list back to the file
            with open(output_file, 'w') as f:
                f.write(str(existing_data))  # Write as a valid Python list
        except Exception as e:
            log(f"Error handling the output file {output_file}: {e}")
            return None

        return distance
    except Exception as e:
        log(f"Error processing {image_path}: {e}")
        return None

def process_images(
    dir_images, 
    dir_references, 
    dir_output, 
    model_class, 
    metric, 
    shape_predictor_68_path, 
    face_recognition_model_path, 
    num_processes
):
    log("Starting image processing...")
    try:
        os.makedirs(dir_output, exist_ok=True)
        ref_embeddings = {}

        # Step 1: Compute embeddings for reference images
        for ref_name in os.listdir(dir_references):
            try:
                ref_path = os.path.join(dir_references, ref_name)
                if ref_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    ref_embedding = get_face_embedding(
                        ref_path, 
                        shape_predictor_68_path,
                        face_recognition_model_path
                    )
                    if ref_embedding is not None:
                        ref_embeddings[ref_name] = ref_embedding
                    else:
                        log(f"Failed to compute embedding for reference image {ref_name}")
            except Exception as e:
                log(f"Failed to process reference image {ref_name}: {e}")
                
        log("=" * 50)
        log(f"Number of reference embeddings: {len(ref_embeddings)}")
        log("=" * 50)

        # Step 2: Create tasks for all image-reference pairs
        tasks = []
        processed_folders = 0
        processed_files = 0
        
        for subdir_name in os.listdir(dir_images):
            subdir_path = os.path.join(dir_images, subdir_name)
            if os.path.isdir(subdir_path):
                output_subdir = os.path.join(dir_output, subdir_name)
                os.makedirs(output_subdir, exist_ok=True)
                processed_folders += 1

                for ref_name, ref_embedding in ref_embeddings.items():
                    image_paths = [
                        os.path.join(subdir_path, img)
                        for img in os.listdir(subdir_path)
                        if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
                    ]
                    tasks.extend([
                        (img_path, ref_embedding, metric, shape_predictor_68_path, face_recognition_model_path, output_subdir, ref_name)
                        for img_path in image_paths
                    ])

        # Step 3: Process tasks in parallel
        with Pool(num_processes, initializer=init_worker, initargs=(shape_predictor_68_path, face_recognition_model_path)) as pool:
            for _ in pool.imap_unordered(process_single_image, tasks):
                processed_files += 1  # Increment count for each processed file

        # Log the results
        log("=" * 50)
        log(f"Total processed folders: {processed_folders}")
        log(f"Total processed TXT files: {processed_files}")
        log("=" * 50)
        log("Image processing completed.")
    except Exception as e:
        log(f"Error during processing: {e}")

# ===== FUNCTION ZOO =======================================

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX [[[ MAIN ]]] XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
def main():
    try:
        # Set up logging with the specified verbosity
        setup_logging()
        log(f"Logging initialized at {LOG_LEVEL} level.")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(script_dir, 'LOGS')
        os.makedirs(logs_dir, exist_ok=True)

        # Validate number of processes
        try:
            num_processes = int(os.getenv('NUM_PROCESSES', 2))
            validate_num_processes(num_processes)
        except ValueError as e:
            log(str(e))
            sys.exit(str(e))

        # Paths to model files for DLib
        dlib_folder = os.path.join(script_dir, 'DLIB')
        validate_dlib_models(dlib_folder)

        dir_images = os.path.join(script_dir, 'DIR', 'images')
        dir_references = os.path.join(script_dir, 'DIR', 'references')
        dir_output = os.path.join(script_dir, 'DIR', 'output')

        shape_predictor_68_path = os.path.join(dlib_folder, 'shape_predictor_68_face_landmarks.dat')
        face_recognition_model_path = os.path.join(dlib_folder, 'dlib_face_recognition_resnet_model_v1.dat')

        if not os.path.exists(dir_images):
            raise FileNotFoundError(f"Images directory not found: {dir_images}")
        if not os.path.exists(dir_references):
            raise FileNotFoundError(f"References directory not found: {dir_references}")
        if not os.path.exists(shape_predictor_68_path):
            raise FileNotFoundError(f"68-point landmark model not found: {shape_predictor_68_path}")
        if not os.path.exists(face_recognition_model_path):
            raise FileNotFoundError(f"Face recognition model not found: {face_recognition_model_path}")

        log(f"Starting script with {num_processes} processes.")

        process_images(
            dir_images,
            dir_references,
            dir_output,
            DLib,
            "L2_norm",
            shape_predictor_68_path,
            face_recognition_model_path,
            num_processes
        )

        log("Script execution completed successfully.")
    except Exception as e:
        logging.error(f"Critical error in main(): {e}")
        sys.exit(1)

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX [[[ MAIN ]]] XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

if __name__ == "__main__":
    main()
