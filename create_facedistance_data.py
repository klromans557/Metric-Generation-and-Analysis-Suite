import os
import logging
import threading
import traceback
import time
import sys
import dlib
import numpy as np
import numpy.linalg as la
import multiprocessing
import hashlib
import pickle
from PIL import Image
from scipy.spatial.distance import cosine, euclidean
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
    getattr(logging, level.lower(), logging.info)(message)
    for handler in logging.getLogger().handlers:
        handler.flush()  # Ensure logs are flushed immediately
    
def log_debug(message):
    # Logs a debug message if LOG_LEVEL is set to DEBUG.
    if LOG_LEVEL == "DEBUG":
        log(message, level="DEBUG")

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
            log_debug(f"Stack trace: {traceback.format_exc()}")
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
        dets = _global_dlib.detector(image, 1)
        log(f"Detected {len(dets)} faces in {image_path}.")
        if dets:
            shape = _global_dlib.shape_predictor_68(image, dets[0])
            aligned_image, _ = _global_dlib.align_face(image, shape)  # Align the face
            if aligned_image is not None:
                embedding = _global_dlib.get_embedding(aligned_image)
                return embedding
        log(f"No valid face embedding for image {image_path}.")
        return None
  
    except Exception as e:
        log(f"Error processing {image_path}: {e}")
        return None

def compute_embedding_worker(ref_path, shape_predictor_68_path, face_recognition_model_path):
        # Worker function for computing a single embedding.
        try:
            embedding = get_face_embedding(
                ref_path, shape_predictor_68_path, face_recognition_model_path
            )
            if embedding is not None:
                return ref_path, embedding, None
            else:
                return ref_path, None, f"Failed to compute embedding for {os.path.basename(ref_path)}"
        except Exception as e:
            return ref_path, None, str(e)

# Compute reference embeddings in parallel or sequentially based on the dataset size
def compute_reference_embeddings(ref_paths, shape_predictor_68_path, face_recognition_model_path, num_processes):
    # Computes embeddings for reference images. Uses parallel processing for large datasets.
    if len(ref_paths) <= 5:  # Small dataset: Sequential processing
        log("Processing reference embeddings sequentially (small dataset).", level="DEBUG")
        results = [
            compute_embedding_worker(ref_path, shape_predictor_68_path, face_recognition_model_path)
            for ref_path in ref_paths
        ]
    else:  # Large dataset: Parallel processing
        log(f"Processing reference embeddings in parallel mode with {num_processes} processes.", level="DEBUG")
        with Pool(num_processes) as pool:
            results = pool.starmap(
                compute_embedding_worker,
                [(ref_path, shape_predictor_68_path, face_recognition_model_path) for ref_path in ref_paths]
            )

    embeddings = {}
    failed_files = []
    for ref_path, embedding, error in results:
        if embedding is not None:
            embeddings[os.path.basename(ref_path)] = embedding
        else:
            failed_files.append((os.path.basename(ref_path), error))

    return embeddings, failed_files

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

def compute_folder_hash(folder_path):
    # Generates a hash for the folder based on file names and modification times.
    hash_obj = hashlib.sha256()
    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            hash_obj.update(file_name.encode('utf-8'))
            hash_obj.update(str(os.path.getmtime(file_path)).encode('utf-8'))
    return hash_obj.hexdigest()

def save_reference_embeddings(embeddings, folder_hash, output_path):
    # Saves embeddings and folder hash to a pickle file.
    data = {
        "embeddings": embeddings,
        "folder_hash": folder_hash
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    log("=" * 50)
    log(f"Reference embeddings saved to {output_path}")

def load_reference_embeddings(folder_hash, input_path="CACHE/ref_embeddings.pkl"):
    # Loads embeddings from a pickle file and checks the folder hash.
    if not os.path.exists(input_path):
        return None  # No cached embeddings
    
    try:
        with open(input_path, "rb") as f:
            data = pickle.load(f)
        if data["folder_hash"] == folder_hash:
            log("=" * 50)
            log("Cached reference embeddings match the current folder.")
            log("Using cached data.")
            log("=" * 50)
            return data["embeddings"]
        log("Cached reference embeddings do not match. Recomputing embeddings.")
        return None
    except Exception as e:
        log(f"Error loading cache file {input_path}: {e}", level="ERROR")
        return None

def process_single_image(args):
    """
    Processes a single image, calculates the similarity distance, 
    and writes the result to the corresponding output file.
    
    Args:
        args (tuple): Contains all necessary inputs for processing.
                      (image_path, ref_embedding, metric, shape_predictor_68_path, 
                       face_recognition_model_path, output_subdir, ref_name)
    """
    image_path, ref_embedding, metric, shape_predictor_68_path, face_recognition_model_path, output_subdir, ref_name = args

    try:
        # Step 1: Get the embedding for the generated image
        log_debug(f"Processing image: {image_path}")
        gen_embedding = get_face_embedding(image_path, shape_predictor_68_path, face_recognition_model_path)
        if gen_embedding is None:
            log(f"Failed to compute embedding for generated image {image_path}. Skipping.", level="WARNING")
            return None
        log_debug(f"Generated embedding for {image_path}: {gen_embedding}")

        # Step 2: Calculate the similarity distance
        log_debug(f"Calculating distance using metric: {metric}")
        distance = compute_distance(gen_embedding, ref_embedding, metric)
        
        if distance is None:
            log(f"Failed to compute a valid distance for {image_path} and {ref_name}.", level="WARNING")
            return None
        log(f"Computed distance for {image_path} and {ref_name}: {distance}")
        
        # Step 3: Validate and log the calculated distance
        if not isinstance(distance, (float, int)):
            log(f"Invalid distance value for {image_path}: {distance}. Skipping write operation.", level="WARNING")
            return None
        log(f"Computed distance for {image_path} and {ref_name}: {distance}")

        # Step 4: Prepare the output file
        output_file = os.path.join(output_subdir, f"{ref_name}.txt")
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir, exist_ok=True)
            log(f"Created output subdirectory: {output_subdir}")

        # Step 5: Read, append, and write data
        existing_data = []
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    existing_data = eval(f.read())
                log(f"Loaded existing data from {output_file}: {existing_data}")
            except Exception as e:
                log(f"Failed to read existing data from {output_file}: {e}", level="ERROR")
        
        # Append and write back
        existing_data.append(distance)
        try:
            with open(output_file, 'w') as f:
                f.write(str(existing_data))
            log(f"Successfully wrote distances to {output_file}: {existing_data}")
        except Exception as e:
            log(f"Failed to write distances to {output_file}: {e}", level="ERROR")
            return None
        
        return True

    except Exception as e:
        log(f"Error processing {image_path}: {e}", level="ERROR")
        return None

def process_images_in_batch(batch_args):
    results = []
    for args in batch_args:
        try:
            log_debug(f"Task arguments: {args}")  # Log the task arguments
            result = process_single_image(args)
            if result:
                log(f"Successfully processed: {args[0]}")
            else:
                log(f"Failed processing task: {args[0]}", level="WARNING")
            results.append(result)
        except Exception as e:
            log(f"Batch processing error for {args[0]}: {e}", level="ERROR")
            log_debug(f"Stack trace: {traceback.format_exc()}")
    return results

def process_images(
    dir_images,
    dir_references,
    dir_output,
    model_class,
    shape_predictor_68_path,
    face_recognition_model_path,
    num_processes,
    metric
):
    log("Starting image processing...")
    processed_files = 0
    total_files = 0
    start_time = time.time()
    
    # Validate the metric function before processing tasks
    log_debug(f"Using metric: {metric}")
    if not isinstance(metric, str):
        raise ValueError(f"Invalid metric identifier: {metric}. It should be a string (e.g., 'L2_norm', 'cosine').")
    log_debug(f"Metric identifier validated: {metric}")

    try:
        os.makedirs(dir_output, exist_ok=True)
        ref_embeddings = {}

        # Track successes and failures
        success_count = 0
        failure_count = 0
        failed_files = []

        # Step 1: Compute embeddings for reference images
        log("=" * 50)
        log("Starting reference embedding computation...")
        
        # Define the cache folder and file path
        cache_folder = os.path.join(os.getcwd(), "CACHE")
        os.makedirs(cache_folder, exist_ok=True)
        cache_file_path = os.path.join(cache_folder, "ref_embeddings.pkl")
        
        folder_hash = compute_folder_hash(dir_references)

        # Attempt to load cached embeddings
        ref_embeddings = load_reference_embeddings(folder_hash, cache_file_path)

        if ref_embeddings is None:  # Recompute if cache is missing or outdated
            ref_paths = [
                os.path.join(dir_references, ref_name)
                for ref_name in os.listdir(dir_references)
                if ref_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
            ]
            ref_embeddings, failed_files = compute_reference_embeddings(ref_paths, shape_predictor_68_path, face_recognition_model_path, num_processes)
            save_reference_embeddings(ref_embeddings, folder_hash, cache_file_path)

            # Log summary of results
            success_count = len(ref_embeddings)
            failure_count = len(failed_files)
            log("=" * 50)
            log(f"Reference embedding summary: {success_count} successful, {failure_count} failed.")
            log("=" * 50)

            # Handle failed references
            if failed_files:
                if len(failed_files) > 5:
                    # Save failed file names to 'failed_files.txt' in the LOGS folder
                    failed_files_path = os.path.join(os.getcwd(), 'LOGS', 'failed_files.txt')
                    try:
                        os.makedirs(os.path.dirname(failed_files_path), exist_ok=True)
                        with open(failed_files_path, 'w') as f:
                            f.writelines(f"{name}: {error}\n" for name, error in failed_files)
                        log(f"More than 5 files failed. Full list saved to: {failed_files_path}")
                        log("=" * 50)
                    except Exception as e:
                        log(f"Error writing failed files to {failed_files_path}: {e}", level="ERROR")
                        log("=" * 50)
                else:
                    log(f"Failed files: {', '.join(name for name, _ in failed_files)}")
                    log("=" * 50)

        # Step 2: Create tasks for all image-reference pairs
        tasks = []
        processed_folders = 0  # Tracks the number of folders processed

        for subdir_name in os.listdir(dir_images):
            subdir_path = os.path.join(dir_images, subdir_name)
            if os.path.isdir(subdir_path):
                output_subdir = os.path.join(dir_output, subdir_name)
                os.makedirs(output_subdir, exist_ok=True)
                processed_folders += 1

                # Debug log for folder processing
                log_debug(f"Processing folder: {subdir_path}. Output folder: {output_subdir}.")

                for ref_name, ref_embedding in ref_embeddings.items():
                    image_paths = [
                        os.path.join(subdir_path, img)
                        for img in os.listdir(subdir_path)
                        if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
                    ]

                    # Track and log the tasks being added
                    tasks_before = len(tasks)
                    tasks.extend([
                        (img_path, ref_embedding, metric, shape_predictor_68_path, face_recognition_model_path, output_subdir, ref_name)
                        for img_path in image_paths
                    ])
                    tasks_added = len(tasks) - tasks_before
                    log_debug(f"Prepared {tasks_added} tasks for reference {ref_name} in folder {subdir_path}.")

                    # Update total file count and log progress
                    total_files += len(image_paths)
                    log_debug(f"Total files processed for {subdir_path}: {len(image_paths)}. Current total: {total_files}.")

        # Handle small datasets gracefully
        if total_files <= 5:
            log(f"Small dataset detected ({total_files} files). Running in single-threaded mode.")
            for i, task in enumerate(tasks, start=1):
                process_single_image(task)
                processed_files += 1
                log(f"Processed {processed_files}/{total_files} files.")
            return

        if total_files < num_processes:
            log(f"Small dataset detected: {total_files} files. Adjusting num_processes to {total_files}. Batch size set to 1.")
            num_processes = total_files
            batch_size = 1
        else:
            # Define batch size (number of images per batch)
            batch_size = min(num_processes * 5, 50, total_files)

        # Split tasks into batches
        batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
        log(f"Batch size: {batch_size}, Total batches: {len(batches)}", level="DEBUG")
        log("=" * 50)
        log(f"Total tasks prepared: {len(tasks)}")
        log("=" * 50)

        # Step 3: Process tasks in parallel
        log("Starting image processing tasks...")

        try:
            log_debug(f"Starting multiprocessing with {num_processes} processes...")

            with Pool(num_processes, initializer=init_worker, initargs=(shape_predictor_68_path, face_recognition_model_path)) as pool:
                for batch_index, batch_results in enumerate(pool.imap_unordered(process_images_in_batch, batches), start=1):
                    batch_size = len(batches[batch_index - 1])
                    log(f"Processing batch {batch_index}/{len(batches)}. Batch size: {batch_size}")

                    if not batch_results:
                        log(f"Batch {batch_index} returned no results. Possible error in processing tasks.", level="WARNING")

                    # Process each result in the batch
                    for task_index, result in enumerate(batch_results, start=1):
                        processed_files += 1
                        elapsed_time = time.time() - start_time

                        if result:
                            log_debug(f"Task {task_index} in batch {batch_index} completed successfully.")
                        else:
                            log_debug(f"Task {task_index} in batch {batch_index} failed.")

                        # Log progress at intervals
                        if processed_files % 10 == 0 or processed_files == total_files:
                            log(f"Processed {processed_files}/{total_files} files. Elapsed time: {int(elapsed_time)} seconds.")

                    # Batch completion summary
                    log(f"Batch {batch_index}/{len(batches)} completed. Total processed: {processed_files}/{total_files}.")
        except Exception as e:
            log(f"Critical error during multiprocessing: {e}", level="CRITICAL")
            log_debug(f"Stack trace: {traceback.format_exc()}")
            sys.exit(1)

        # Log the final summary
        log(f"Processed {total_files}/{total_files} files. Total time: {int(time.time() - start_time)} seconds.")

        # Log the results
        log("=" * 50)
        log(f"Total processed folders: {processed_folders}")
        log(f"Completed {len(tasks)} files in {int(time.time() - start_time)} seconds.")
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
            log(f"Configuration error: {e}. Check the number of processes.", level="ERROR")
            log_debug(f"Stack trace: {traceback.format_exc()}")
            sys.exit(1)

        # Paths to model files for DLib
        dlib_folder = os.path.join(script_dir, 'DLIB')
        try:
            validate_dlib_models(dlib_folder)
        except FileNotFoundError as e:
            log(f"Critical error: {e}. Ensure all required DLib models are present.", level="CRITICAL")
            log_debug(f"Stack trace: {traceback.format_exc()}")
            sys.exit(1)

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

        try:
            process_images(
                dir_images,
                dir_references,
                dir_output,
                DLib,
                shape_predictor_68_path,
                face_recognition_model_path,
                num_processes,
                metric="L2_norm"  # Keyword argument
            )

            log("Script execution completed successfully.")
            
        except FileNotFoundError as e:
            log(f"File or directory error: {e}. Ensure all input files and directories are correctly specified.", level="CRITICAL")
            log_debug(f"Stack trace: {traceback.format_exc()}")
            sys.exit(1)
        except ValueError as e:
            log(f"Configuration error: {e}. Check the input parameters and try again.", level="ERROR")
            log_debug(f"Stack trace: {traceback.format_exc()}")
        except MemoryError:
            log("Memory error detected. Try reducing batch size or num_processes.", level="ERROR")
            log_debug(f"Stack trace: {traceback.format_exc()}")
            sys.exit("Script terminated due to insufficient memory.")
        except Exception as e:
            log(f"Unexpected error: {e}. Please check the log file for more information.", level="CRITICAL")
            log_debug(f"Stack trace: {traceback.format_exc()}")
            sys.exit(1)
        
    except Exception as e:
        logging.error(f"Critical error in main(): {e}")
        sys.exit(1)

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX [[[ MAIN ]]] XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

if __name__ == "__main__":
    main()
