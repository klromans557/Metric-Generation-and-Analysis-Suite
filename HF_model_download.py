import os
from huggingface_hub import hf_hub_download

def download_file(repo_id, filename, local_dir, rename_as=None):
    local_path = os.path.join(local_dir, rename_as or filename)
    if not os.path.exists(local_path):
        try:
            print(f"Downloading {filename} from {repo_id}...")
            downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
            if rename_as:
                os.rename(downloaded_path, local_path)
            print(f"{filename} downloaded successfully as {rename_as or filename}.")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
    else:
        print(f"{rename_as or filename} already exists.")

def main():
    dlib_repo = "matt3ounstable/dlib_predictor_recognition"
    clip_repo = "Qdrant/clip-ViT-B-32-vision"
    local_dir = "./DLIB"

    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    download_file(dlib_repo, "shape_predictor_68_face_landmarks.dat", local_dir)
    download_file(dlib_repo, "dlib_face_recognition_resnet_model_v1.dat", local_dir)
    download_file(clip_repo, "model.onnx", local_dir, rename_as="clip-ViT-B-32-vision.onnx")

if __name__ == "__main__":
    main()
