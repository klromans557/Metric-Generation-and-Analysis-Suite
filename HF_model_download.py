import os
from huggingface_hub import hf_hub_download

def download_file(repo_id, filename, local_dir):
    local_path = os.path.join(local_dir, filename)
    if not os.path.exists(local_path):
        try:
            print(f"Downloading {filename}...")
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
            print(f"{filename} downloaded successfully.")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
    else:
        print(f"{filename} already exists.")

def main():
    repo_id = "matt3ounstable/dlib_predictor_recognition"
    local_dir = "./DLIB"
    
    # Ensure the directory exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    download_file(repo_id, "shape_predictor_68_face_landmarks.dat", local_dir)
    download_file(repo_id, "dlib_face_recognition_resnet_model_v1.dat", local_dir)

if __name__ == "__main__":
    main()