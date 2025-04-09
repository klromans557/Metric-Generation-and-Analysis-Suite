import os
import sys
import traceback
import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.spatial.distance import cosine
import hashlib
import pickle
from datetime import datetime
from multiprocessing import Pool, cpu_count, Manager
import ast

# Configuration
DISTANCE_METRIC = 'cosine'  # Options: 'l2', 'cosine'
NUM_PROCESSES = int(os.getenv('NUM_PROCESSES', min(4, (os.cpu_count() // 2) + 1)))

# Paths
DIR = os.path.join(os.getcwd(), 'DIR')
IMAGES_DIR = os.getenv('IMAGES_DIR', os.path.join(os.getcwd(), 'DIR', 'images'))
REFERENCES_DIR = os.getenv('REFERENCES_DIR', os.path.join(os.getcwd(), 'DIR', 'references'))
OUTPUT_DIR = os.getenv('OUTPUT_DIR', os.path.join(os.getcwd(), 'DIR', 'output'))
LOGS_DIR = os.getenv('LOGS_DIR', os.path.join(os.getcwd(), 'LOGS'))
CACHE_DIR = os.path.join(os.getcwd(), 'CACHE')
STYLE_LOG_PATH = os.path.join(LOGS_DIR, 'styledata_log.txt')
STYLE_CACHE_PATH = os.path.join(CACHE_DIR, 'ref_embeddings_style.pkl')
MODEL_PATH = os.path.join("DLIB", "clip-ViT-B-32-vision.onnx")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Logging
LOG_LEVEL = "INFO"

def setup_logging():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(STYLE_LOG_PATH, 'a') as f:
        f.write(f"\n--- Log started at {timestamp} ---\n")

def log(message, level="INFO"):
    print(f"{level}: {message}")
    with open(STYLE_LOG_PATH, 'a') as f:
        f.write(f"{level}: {message}\n")

def log_debug(message):
    if LOG_LEVEL == "DEBUG":
        log(message, level="DEBUG")

# Utilities
def compute_folder_hash(folder_path):
    hash_obj = hashlib.sha256()
    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            hash_obj.update(file_name.encode('utf-8'))
            hash_obj.update(str(os.path.getmtime(file_path)).encode('utf-8'))
    return hash_obj.hexdigest()

def load_reference_embeddings(folder_hash):
    if not os.path.exists(STYLE_CACHE_PATH):
        return None
    try:
        with open(STYLE_CACHE_PATH, 'rb') as f:
            data = pickle.load(f)
        if data['folder_hash'] == folder_hash:
            log("Using cached reference embeddings.")
            return data['embeddings']
    except Exception as e:
        log(f"Error loading cached embeddings: {e}", level="ERROR")
    return None

def save_reference_embeddings(embeddings, folder_hash):
    with open(STYLE_CACHE_PATH, 'wb') as f:
        pickle.dump({"embeddings": embeddings, "folder_hash": folder_hash}, f)
    log("Reference embeddings cached.")

# Global for multiprocessing
_global_clip_session = None
_locks = None

def init_worker(model_path, locks):
    global _global_clip_session, _locks
    if _global_clip_session is None:
        session_opts = ort.SessionOptions()
        session_opts.intra_op_num_threads = 1
        _global_clip_session = ort.InferenceSession(model_path, sess_options=session_opts)
    _locks = locks

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

def get_clip_embedding(image_path):
    try:
        img = preprocess_image(image_path)
        ort_inputs = {_global_clip_session.get_inputs()[0].name: img}
        output = _global_clip_session.run(None, ort_inputs)[0][0]
        return output / np.linalg.norm(output)
    except Exception as e:
        log(f"Failed to embed {image_path}: {e}", level="ERROR")
        return None

def compute_distance(embedding1, embedding2, metric):
    if metric == 'l2':
        return np.linalg.norm(embedding1 - embedding2) / 2.0
    elif metric == 'cosine':
        return cosine(embedding1, embedding2)
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")

def process_single_image(args):
    gen_image_path, ref_name, ref_emb, output_subdir, metric = args
    try:
        gen_embedding = get_clip_embedding(gen_image_path)
        if gen_embedding is None:
            return (gen_image_path, False)
        distance = compute_distance(gen_embedding, ref_emb, metric)

        output_file_path = os.path.join(output_subdir, f"{ref_name}.txt")
        lock = _locks[ref_name]

        with lock:
            try:
                if os.path.exists(output_file_path):
                    with open(output_file_path, 'r') as f:
                        existing_data = ast.literal_eval(f.read())
                else:
                    existing_data = []
            except Exception as e:
                log(f"Warning: Corrupted or unreadable output file {output_file_path}: {e}", level="WARNING")
                existing_data = []

            existing_data.append(float(distance))

            with open(output_file_path, 'w') as f:
                f.write(str(existing_data))

        return (gen_image_path, True)

    except Exception as e:
        log(f"Error processing {gen_image_path}: {e}", level="ERROR")
        return (gen_image_path, False)

def main():
    try:
        setup_logging()
        log("Starting styledata embedding process.")

        failed_files = []
        folder_hash = compute_folder_hash(REFERENCES_DIR)
        ref_embeddings = load_reference_embeddings(folder_hash)

        if ref_embeddings is None:
            ref_embeddings = {}
            log("Generating reference embeddings...")
            for ref_image in os.listdir(REFERENCES_DIR):
                ref_path = os.path.join(REFERENCES_DIR, ref_image)
                if ref_image.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    session = ort.InferenceSession(MODEL_PATH)
                    img = preprocess_image(ref_path)
                    ort_inputs = {session.get_inputs()[0].name: img}
                    emb = session.run(None, ort_inputs)[0][0]
                    emb /= np.linalg.norm(emb)
                    ref_embeddings[ref_image] = emb
            save_reference_embeddings(ref_embeddings, folder_hash)

        tasks = []
        for subdir_name in os.listdir(IMAGES_DIR):
            subdir_path = os.path.join(IMAGES_DIR, subdir_name)
            if not os.path.isdir(subdir_path):
                continue

            output_subdir = os.path.join(OUTPUT_DIR, subdir_name)
            os.makedirs(output_subdir, exist_ok=True)

            for gen_image_name in os.listdir(subdir_path):
                gen_image_path = os.path.join(subdir_path, gen_image_name)
                if gen_image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    for ref_name, ref_emb in ref_embeddings.items():
                        tasks.append((gen_image_path, ref_name, ref_emb, output_subdir, DISTANCE_METRIC))

        log(f"Prepared {len(tasks)} tasks for processing.")

        if len(tasks) <= 10:
            log("Small dataset detected. Running in single-threaded mode.")
            for task in tasks:
                process_single_image(task)
        else:
            with Manager() as manager:
                locks = manager.dict({ref_name: manager.Lock() for ref_name in ref_embeddings})
                with Pool(NUM_PROCESSES, initializer=init_worker, initargs=(MODEL_PATH, locks)) as pool:
                    results = pool.map(process_single_image, tasks)
                    failed_files.extend([path for path, success in results if not success])

        completed = len(tasks) - len(failed_files)
        log(f"Summary: {completed} tasks completed successfully.")
        log(f"Summary: {len(failed_files)} tasks failed.")

        if failed_files:
            failed_path = os.path.join(LOGS_DIR, 'failed_files_style.txt')
            with open(failed_path, 'w') as f:
                f.writelines(f"{path}\n" for path in failed_files)
            log(f"Failed image paths written to: {failed_path}")

        # if ref_embeddings:
            # log(f"Embedding dimension: {next(iter(ref_embeddings.values())).shape[0]}")
        log(f"Style distance data creation complete using '{DISTANCE_METRIC}' distance.")

    except Exception as e:
        log(f"Fatal error: {e}", level="CRITICAL")
        log_debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()