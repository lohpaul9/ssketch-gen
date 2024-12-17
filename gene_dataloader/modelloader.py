from huggingface_hub import HfApi, HfFolder, Repository
import os
import shutil

def sendModelToHuggingFace(controlnet_image, controlnet_sketch, model):
    # Verify paths
    if not os.path.exists(controlnet_image) or not os.path.exists(controlnet_sketch):
        raise FileNotFoundError("Model files not found at the specified paths.")
    
    # Clone repository or create a new one
    repo_dir = "./temp_model_repo"
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)  # Clean up any previous directory
    
    print("Cloning or creating Hugging Face repository...")
    repo = Repository(local_dir=repo_dir, clone_from=model, use_auth_token=True)

    # Copy model files into the repository
    shutil.copy(controlnet_image, os.path.join(repo_dir, "controlnet_image"))
    shutil.copy(controlnet_sketch, os.path.join(repo_dir, "controlnet_sketch"))

    print("Adding and pushing files to Hugging Face Hub...")
    repo.git_add(auto_lfs_tracking=True)  # Auto-track large files with LFS
    repo.git_commit("Add ControlNet models: image and sketch")
    repo.git_push()
    print(f"Models successfully uploaded to https://huggingface.co/{model}")

model = "genecodes/shadystrokes"
controlnet_image = "/home/ec2-user/diffusers-sketchy/examples/controlnet/output_model/controlnet_image"
controlnet_sketch = "/home/ec2-user/diffusers-sketchy/examples/controlnet/output_model/controlnet_sketch"
sendModelToHuggingFace(controlnet_image, controlnet_sketch, model)