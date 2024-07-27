import os
import logging
import sys
import numpy as np
from PIL import Image
import dlib
from scipy.spatial.distance import cosine, euclidean
import numpy.linalg as la
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
def log(message, log_file):
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def is_directory_empty(directory):
    return not any(os.scandir(directory))
# ===== LOGGING ============================================

# ===== CLASSES ============================================
class DLib:
    def __init__(self, shape_predictor_path, face_recognition_model_path):
        self.detector = dlib.get_frontal_face_detector()
        if not os.path.exists(shape_predictor_path):
            raise FileNotFoundError(f"The file {shape_predictor_path} does not exist.")
        if not os.path.exists(face_recognition_model_path):
            raise FileNotFoundError(f"The file {face_recognition_model_path} does not exist.")
        self.shape_predictor_path = shape_predictor_path
        self.face_recognition_model_path = face_recognition_model_path
        self.shape_predictor = dlib.shape_predictor(self.shape_predictor_path)
        self.face_rec_model = dlib.face_recognition_model_v1(self.face_recognition_model_path)

    def get_embedding(self, image):
        dets = self.detector(image, 1)
        if dets:
            shape = self.shape_predictor(image, dets[0])
            embedding = np.array(self.face_rec_model.compute_face_descriptor(image, shape))
            norm = np.linalg.norm(embedding)
            if norm == 0:
                return embedding
            return embedding / norm
        return None
# ===== CLASSES ============================================

# ===== FUNCTION ZOO =======================================
def get_face_embedding(image_path, model_class, shape_predictor_path, face_recognition_model_path):
    model = model_class(shape_predictor_path, face_recognition_model_path)
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    embedding = model.get_embedding(image)
    return embedding

def compute_distance(embedding1, embedding2, metric):
    if metric == "cosine":
        return cosine(embedding1, embedding2)
    elif metric == "euclidean":
        return euclidean(embedding1, embedding2)
    elif metric == "L2_norm":
        return la.norm(embedding1 - embedding2)
    else:
        raise ValueError("Unsupported distance metric")

def process_single_image(args):
    image_path, ref_embedding, metric, model_class, shape_predictor_path, face_recognition_model_path = args
    embedding = get_face_embedding(image_path, model_class, shape_predictor_path, face_recognition_model_path)
    if embedding is not None:
        return compute_distance(ref_embedding, embedding, metric)
    return None

# ============================================================
def process_images(dir_images, dir_references, dir_output, model_class, metric, shape_predictor_path, face_recognition_model_path, num_processes, log_file):
    log("Starting the image processing...", log_file)

    try:
        if not os.path.exists(dir_output):
            os.makedirs(dir_output)

        for subdir_name in os.listdir(dir_images):
            subdir_path = os.path.join(dir_images, subdir_name)
            if os.path.isdir(subdir_path):
                subdir_output_path = os.path.join(dir_output, subdir_name)
                if not os.path.exists(subdir_output_path):
                    os.makedirs(subdir_output_path)
                output_files_generated = 0

                for ref_image_name in os.listdir(dir_references):
                    log(f"Start reference: {ref_image_name}; folder: {subdir_name}", log_file)

                    ref_image_path = os.path.join(dir_references, ref_image_name)
                    ref_embedding = get_face_embedding(ref_image_path, model_class, shape_predictor_path, face_recognition_model_path)
                    if ref_embedding is None:
                        continue

                    image_paths = [os.path.join(subdir_path, image_name) for image_name in os.listdir(subdir_path)]
                    distances = []

                    with Pool(processes=num_processes) as pool:
                        args = [(image_path, ref_embedding, metric, model_class, shape_predictor_path, face_recognition_model_path) for image_path in image_paths]
                        results = pool.map(process_single_image, args)
                        distances = [result for result in results if result is not None]

                    output_file = os.path.join(subdir_output_path, f"{ref_image_name.split('.')[0]}_distances.txt")
                    with open(output_file, 'w') as f:
                        f.write(str(distances))
                    output_files_generated += 1

                    log(f"End reference: {ref_image_name}; folder: {subdir_name}", log_file)

                log(f"{'=' * 50}\n"f"Finished processing all reference images for folder: {subdir_name}.\n"f"Generated {output_files_generated} output TXT files.\n"f"{'=' * 50}", log_file)

    except Exception as e:
        error_message = f"An error occurred during image processing: {str(e)}"
        log(error_message, log_file)
        sys.exit(error_message)

if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(script_dir, 'LOGS')
        
       # Ensure the LOGS directory exists
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        log_file = os.path.join(logs_dir, 'process_log.txt')
        
        # Clear the log file at the start of each run
        with open(log_file, 'w') as f:
            f.write('Starting new run...\n')

        # Paths to model files for DLib; place these .dat files in the DLIB directory in order for the script to find them
        # ¡NOTE! Use these HuggingFace links: (https://huggingface.co/matt3ounstable/dlib_predictor_recognition/resolve/main/shape_predictor_5_face_landmarks.dat?download=true) 
        # ¡NOTE! and (https://huggingface.co/matt3ounstable/dlib_predictor_recognition/resolve/main/dlib_face_recognition_resnet_model_v1.dat?download=true)
        # Thanks to Matteo Spinelli (a.k.a cubiq) for hosting the above files on HF, so that I can continue to be lazy! :)
        dlib_folder = os.path.join(script_dir, 'DLIB')
        shape_predictor_path = os.path.join(dlib_folder, 'shape_predictor_5_face_landmarks.dat')
        face_recognition_model_path = os.path.join(dlib_folder, 'dlib_face_recognition_resnet_model_v1.dat')

        # Read directories and options from environment variables or use defaults
        dir_images = os.getenv('IMAGES_DIR', os.path.join(os.getcwd(), 'DIR', 'images'))
        dir_references = os.getenv('REFERENCES_DIR', os.path.join(os.getcwd(), 'DIR', 'references'))
        dir_output = os.getenv('OUTPUT_DIR', os.path.join(os.getcwd(), 'DIR', 'output'))
        num_processes = int(os.getenv('NUM_PROCESSES', 2))

        print(f"IMAGES_DIR: {os.getenv('IMAGES_DIR')}")
        print(f"REFERENCES_DIR: {os.getenv('REFERENCES_DIR')}")
        print(f"OUTPUT_DIR: {os.getenv('OUTPUT_DIR')}")
        print(f"GRAPH_WIDTH: {os.getenv('GRAPH_WIDTH')}")
        print(f"GRAPH_HEIGHT: {os.getenv('GRAPH_HEIGHT')}")
        print(f"NUM_PROCESSES: {os.getenv('NUM_PROCESSES')}")

        # Check if directories are empty and log errors
        if is_directory_empty(dlib_folder):
            error_message = "Error: The DLIB directory is empty. Please add the .dat files."
            log(error_message, log_file)
            sys.exit(error_message)

        if is_directory_empty(dir_images):
            error_message = "Error: The images directory is empty."
            log(error_message, log_file)
            sys.exit(error_message)

        if is_directory_empty(dir_references):
            error_message = "Error: The references directory is empty."
            log(error_message, log_file)
            sys.exit(error_message)

        # Define the distance metric to be used
        distance_metric = "L2_norm"  # Default: "L2_norm"; Options: "cosine", "euclidean", "L2_norm"

        # Do stuff!
        model_class = DLib
        log("Using DLib model.", log_file)
        log(f"Script started using CPU.", log_file)
        process_images(dir_images, dir_references, dir_output, model_class, distance_metric, shape_predictor_path, face_recognition_model_path, num_processes, log_file)
        log("Script completed successfully.", log_file)
    except Exception as e:
        error_message = f"An error occurred in the main function: {str(e)}"
        log(error_message, log_file)
        sys.exit(error_message)
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
