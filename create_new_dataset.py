from datasets import load_dataset
import numpy as np
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
from pathlib import Path
import io
from models.pix2pix_model import Pix2PixModel
from argparse import Namespace

def create_difference_drawing_dataset():
    # Create output directories
    output_dir = "./pipe_drawings_dataset"
    
    ## ----------------- LOADING IMAGES ----------------- ##

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "first_images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "second_images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "drawings"), exist_ok=True)

    # Get cached dataset path
    cache_dir = Path.home() / ".cache/huggingface/hub/datasets--paint-by-inpaint--PIPE"
    if not cache_dir.exists():
        print("Dataset not found in cache. Loading dataset first time...")
        dataset = load_dataset("paint-by-inpaint/PIPE", "default")
        print("Dataset cached")
        
    # Look in snapshots directory for the parquet files
    snapshot_dir = next(cache_dir.glob("snapshots/*"))
    print(f"snapshot_dir: {snapshot_dir}")
    # Get the first parquet file from the data directory
    cached_file = next((snapshot_dir / "data").glob("train-*.parquet"))
    print(f"Using parquet file: {cached_file}")
    
    # Load directly from cached parquet files
    import pyarrow.parquet as pq
    table = pq.read_table(cached_file)
    data = table.to_pandas()

    ## ----------------- LOADING MASKS ----------------- ##

    # get cached masks path
    mask_cache_dir = Path.home() / ".cache/huggingface/hub/datasets--paint-by-inpaint--PIPE_Masks"
    if not mask_cache_dir.exists():
        mask_data_files = {"train": "data/train-*", "test": "data/test-*"}
        dataset_masks  = load_dataset('paint-by-inpaint/PIPE_Masks',data_files=mask_data_files)
        print("Dataset cached")
        
    # Look in snapshots directory for the parquet files
    mask_snapshot_dir = next(mask_cache_dir.glob("snapshots/*"))
    print(f"snapshot_dir: {mask_snapshot_dir}")
    # Get the first parquet file from the data directory
    mask_cached_file = next((mask_snapshot_dir / "data").glob("train-*.parquet"))
    print(f"Using parquet file: {mask_cached_file}")
    
    # Load directly from cached parquet files
    import pyarrow.parquet as pq
    mask_table = pq.read_table(mask_cached_file)
    mask_data = mask_table.to_pandas()

    ## ----------------- LOADING PRETRAINED PHOTSKETCH MODEL ----------------- ##

    # Create model and initialize it with required options
    opt = Namespace(
        isTrain=False,
        input_nc=3,  # input channels
        output_nc=1,  # output channels (1 for sketch)
        ngf=64,      # number of gen filters
        which_model_netG='resnet_9blocks',
        norm='batch',
        no_dropout=True,
        init_type='normal',
        which_direction='AtoB',  # direction of conversion
        use_cuda = False,
        save_dir = '/pipe_drawings_dataset/drawing',
        checkpoints_dir = '/pipe_drawings_dataset/checkpoints',
        name = 'pipe_drawings_dataset',
        results_dir = '/pipe_drawings_dataset/results',
        check_point_path = '/pipe_drawings_dataset/checkpoints/latest_net_G.pth',
        which_epoch = 'latest',
        pretrain_path = './pretrained_model'
    )
    
    model = Pix2PixModel()
    model.initialize(opt)

    # img_transforms = transforms.Compose([transforms.Resize(256, Image.BICUBIC),
    #                 transforms.ToTensor()]) 

    ## ---------------------------- BATCH GENERATION -------------------- ##
    
    BATCH_SIZE = 2  # You can adjust this value
    
    # Process in batches
    for batch_idx in range(0, len(data), BATCH_SIZE):
        batch_end = min(batch_idx + BATCH_SIZE, len(data))
        batch_data = data[batch_idx:batch_end]
        batch_mask_data = mask_data[batch_idx:batch_end]

        print(f"batch_data shape: {batch_data.shape}, batch_mask_data shape: {batch_mask_data.shape}")

        # Collect images and masks for the batch
        images = []
        masks = []

        for idx, row in batch_data.iterrows():
            # Get the second image (target image)
            second_image = Image.open(io.BytesIO(row['target_img']['bytes']))

            print(f"second_image shape: {second_image.size}")
            
            # Transform image - remove the unsqueeze(0) since we'll stack them later
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            transformed_image = transform(second_image)  # Remove unsqueeze(0)

            print(f"transformed_image shape: {transformed_image.shape}")
            
            images.append(transformed_image)
            
        for idx, row in batch_mask_data.iterrows():
            # import ipdb; ipdb.set_trace()
            mask = Image.open(io.BytesIO(row['mask']['bytes']))

            # Get target size from the corresponding image
            target_size = (512, 512)  # Or get from second_image.size
            
            # Resize mask if dimensions don't match
            if mask.size != target_size:
                print(f"Resizing mask from {mask.size} to {target_size}")
                mask = mask.resize(target_size, Image.NEAREST)  # NEAREST to preserve binary values
                
            # convert to tensor
            mask = transforms.ToTensor()(mask)

            print(f"mask shape: {mask.shape}")

            masks.append(mask)
        
        # Stack images and masks
        batch_images = torch.stack(images, dim=0)
        batch_masks = np.stack(masks, axis=0)
        
        # Prepare model input
        model_input = {
            'A': batch_images,
            'B': batch_images,
            'A_paths': [''] * len(batch_images),
            'B_paths': [''] * len(batch_images)
        }
        
        # Process batch through model
        model.set_input(model_input)
        model.test()  # Forward pass
        sketches = model.get_batch_outputs()
        print(f"sketches shape: {sketches.shape}")
        
        # Apply masks to sketches
        for i, (sketch, mask) in enumerate(zip(sketches, batch_masks)):
            # convert sketch to numpy
            sketch = sketch.detach().cpu().numpy()

            print(f"mask shape: {mask.shape}, sketch shape: {sketch.shape}")

            # Remove extra dimensions to get (H, W)
            sketch = np.squeeze(sketch)  
            mask = np.squeeze(mask)  
            
            # Convert from [-1, 1] to [0, 255]
            sketch = ((sketch + 1) * 127.5).astype(np.uint8)
            
            # Apply mask
            sketch[mask <= 0.5] = 255   # Black where mask is inactive
            
            print(f"sketch shape after mask application: {sketch.shape}")
            print(f"Sketch value range: [{sketch.min()}, {sketch.max()}]")
            
            # Save individual sketch
            sketch_img = Image.fromarray(sketch, mode='L')
            sketch_img.save(os.path.join(output_dir, "drawings", f"drawing_{batch_idx + i}.png"))

            print(f"sketch img output: {sketch_img}")

            print(f"Saved sketch {batch_idx + i}.png")
        
        if batch_idx % 100 == 0:
            print(f"Processed {batch_idx} images")

if __name__ == "__main__":
    create_difference_drawing_dataset()