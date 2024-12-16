import json
import os

import sys
import io
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from datasets import Dataset

from models.pix2pix_model import Pix2PixModel
from argparse import Namespace

from datasets import load_dataset
from pprint import pprint

IMG_SIZE = 512

img_transforms = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE), Image.BICUBIC),
            transforms.ToTensor()])

def read_arrow_file(file_path):
    ds = Dataset.from_file(file_path.as_posix())
    ds_pandas = ds.to_pandas()
    return ds_pandas

huggingface_cache_dir = Path("/bigdrive/huggingface")
target_dir = Path("/bigdrive/datasets/sketchy2pix/final")
# input_images_dir = huggingface_cache_dir / Path("paint-by-inpaint___pipe/default-6b5ef3f90595fb74/0.0.0/fb1d63c1965548f86e7309c1fe280556685312e2")
input_images_dir = huggingface_cache_dir / "random_pipe_subset"
input_mask_dir = huggingface_cache_dir / Path("paint-by-inpaint___pipe_masks/default-f6be8545f0afaff5/0.0.0/5bb4a8fbe514cb81618cfef357baa5d9ac9ec21c")


def copy_random_files(source_dir, target_dir, num_files):
    import random
    import shutil
    os.makedirs(target_dir, exist_ok=True)
    source_files = list(source_dir.glob("*.arrow"))
    # remove *test.arrow
    source_files = [file for file in source_files if "test" not in file.name]
    random_files = random.sample(source_files, num_files)
    i = 12
    for file in random_files:
        print(f"Copying {file} to {target_dir / f'{i}.arrow'}")
        shutil.copy(file, target_dir / f"{i}.arrow")
        i += 1

# copy_target = huggingface_cache_dir / "random_pipe_subset"
# copy_random_files(input_images_dir, copy_target, 12)

def get_memory_map_for_masks():
    source_files = list(input_mask_dir.glob("*.arrow"))
    source_files = [file for file in source_files if "test" not in file.name]

    # create one huge memory map of img_id to mask
    img_id_to_mask = {}
    for file in source_files:
        data = read_arrow_file(file)
        for idx, row in data.iterrows():
            img_id_to_mask[str(row['img_id'])] = row['mask']
        

    return img_id_to_mask

def get_net_G():
    sys.path.append('./informative-drawings')
    from model import Generator

    opt = {
        "input_nc": 3,
        "output_nc": 1,
        "n_blocks": 3,
        "checkpoints_dir": "./informative-drawings/models",
        "name": "anime_style",
        "which_epoch": "latest",
        "size": 512,
    }

    with torch.no_grad():
        net_G = 0
        net_G = Generator(opt['input_nc'], opt['output_nc'], opt['n_blocks'])
        net_G.cuda()

        net_G.load_state_dict(torch.load(os.path.join(opt['checkpoints_dir'], opt['name'], f'netG_A_{opt["which_epoch"]}.pth')))
        print('loaded', os.path.join(opt['checkpoints_dir'], opt['name'], f'netG_A_{opt["which_epoch"]}.pth'))

        net_G.eval()

    sys.path.remove('./informative-drawings')

    return net_G


def get_pix2pix_model():
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
        use_cuda = True,
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

    return model


def generate_subset_dataset_for_file(input_image_file_path, img_id_to_mask, net_G, pix2pix_model, is_informative_drawings):
    if not (input_image_file_path).exists():
        raise ValueError(f"File {input_images_dir / input_image_file_path} does not exist")
    
    input_image_data = read_arrow_file(input_image_file_path)
    
    # instantiate informative-drawings model
    sys.path.append('./informative-drawings')

    
    # hyperparams for dataset generation
    BATCH_SIZE = 20  #cannot be more than 40?

    # batched dataset generation
    num_batches = min(len(input_image_data) // BATCH_SIZE + 1, 500)
    for batch in range(num_batches):
        start_idx = batch*BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE

        targets, masks, img_ids = [], [], []
        for idx, row in input_image_data[start_idx:end_idx].iterrows():
            img_id = str(row['img_id'])
            img_ids.append(img_id)
            dir_to_store = target_dir / f"{img_id}"
            os.makedirs(dir_to_store, exist_ok=True)

            src_img = Image.open(io.BytesIO(row['source_img']['bytes']))
            src_img.save(dir_to_store / f"{img_id}_0.png")
            src_img = img_transforms(src_img)

            tgt_img = Image.open(io.BytesIO(row['target_img']['bytes']))
            tgt_img.save(dir_to_store / f"{img_id}_1.png")
            tgt_img = img_transforms(tgt_img)
            targets.append(tgt_img)

            prompt = row['Instruction_VLM-LLM']
            # store it in prompt.json as the field "edit"
            with open(dir_to_store / "prompt.json", "w") as f:
                json.dump({"edit": prompt, "created_with": "informative-drawings" if is_informative_drawings else "pix2pix"}, f)

            mask = img_id_to_mask[img_id]
            mask = Image.open(io.BytesIO(mask['bytes']))
            if mask.size != (IMG_SIZE, IMG_SIZE):
                mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
            mask.save(dir_to_store / f"{img_id}_2.png")

            # convert mask to tensor    
            mask = img_transforms(mask)
            masks.append(mask)
        
        assert len(targets) == len(masks)

        if is_informative_drawings:

            targets = torch.stack(targets).to('cuda')
            masks = np.stack(masks, axis=0)
            res = net_G(targets)
            res = res.detach().cpu().numpy()
            res = res.squeeze(1)
            res[masks < 0.5] = 1
            res = (res > 0.7).astype(np.uint8)

            out = res * 255
            for i in range(len(targets)):
                dir_to_store = target_dir / f"{img_ids[i]}"
                masked_image = Image.fromarray(out[i])
                masked_image.save(dir_to_store / f"{img_ids[i]}_3.png")
        else:
            # Stack images and masks
            batch_images = torch.stack(targets, dim=0).to('cuda')
            batch_masks = np.stack(masks, axis=0)
            
            # Prepare model input
            model_input = {
                'A': batch_images,
                'B': batch_images,
                'A_paths': [''] * len(batch_images),
                'B_paths': [''] * len(batch_images)
            }

            pix2pix_model.set_input(model_input)
            pix2pix_model.test()  # Forward pass
            sketches =  pix2pix_model.get_batch_outputs()
            # Apply masks to sketches
        for i, (sketch, mask) in enumerate(zip(sketches, batch_masks)):
            # convert sketch to numpy
            sketch = sketch.detach().cpu().numpy()

            # Remove extra dimensions to get (H, W)
            sketch = np.squeeze(sketch)  
            mask = np.squeeze(mask)  
            
            # Convert from [-1, 1] to [0, 255]
            sketch = ((sketch + 1) * 127.5).astype(np.uint8)
            
            # Apply mask
            sketch[mask <= 0.5] = 255   # Black where mask is inactive

            # binarize sketch
            threshold = 128  
            sketch = np.where(sketch > threshold, 255, 0).astype(np.uint8)
            
            # Save individual sketch
            sketch_img = Image.fromarray(sketch, mode='L')
            dir_to_store = target_dir / f"{img_ids[i]}"
            sketch_img.save(dir_to_store / f"{img_ids[i]}_3.png")

        print(f"Finished storing sketches for batch {batch} of size {len(targets)}")

        if len(targets) < 20:
            break


def set_latest_file_processed(dir, number):
    # write to a checkpoint.json file in the directory
    with open(dir / "checkpoint.json", "w") as f:
        json.dump({"latest_file": number}, f)

def get_latest_file_processed(dir):
    # read the checkpoint.json file in the directory
    with open(dir / "checkpoint.json", "r") as f:
        return json.load(f)["latest_file"]

def generate_entire_subset_dataset(i=None, i_end=10000000):
    os.makedirs(target_dir, exist_ok=True)

    img_id_to_mask = get_memory_map_for_masks()
    net_G = get_net_G()
    pix2pix_model = get_pix2pix_model()
    latest_file_processed = get_latest_file_processed(target_dir)
    print(f"Latest file processed: {latest_file_processed}")

    list_of_files = input_images_dir.glob("*.arrow")
    print(list_of_files)
    # sort by name, files are 1.arrow, 2.arrow, etc.
    list_of_files = list(sorted(list_of_files, key=lambda x: int(x.name.split(".")[0])))

    if i is None:
        i = latest_file_processed + 1

    for i in range(i, min(i_end, len(list_of_files))):
        file = list_of_files[i]
        print(f"Processing {file}")
        generate_subset_dataset_for_file(file, img_id_to_mask, net_G, pix2pix_model, is_informative_drawings=False)
        print(f"----------------Finished processing {file}----------------")
        set_latest_file_processed(target_dir, i)
        i += 1
    
    print(f"Finished processing {i} files")


def repopulate_json_files(file_number):
    file = input_images_dir / f"{file_number}.arrow"
    rows = read_arrow_file(file)
    print(f"Read {len(rows)} rows from {file}")

    errors = []
    
    for idx, row in rows.iterrows():
        prompt = row['Instruction_VLM-LLM']
        if not prompt:
            prompt = row['Instruction_Ref_Dataset']
        json_item = {
            "edit": prompt,
            "img_id": row['img_id'],
            "target_img_dataset": row['target_img_dataset'],
            "Instruction_Ref_Dataset": row['Instruction_Ref_Dataset'],
            "ann_id": row['ann_id'],
        }

        # first get the json file, to see whether it was created with pix2pix or informative-drawings
        json_file = target_dir / f"{row['img_id']}" / "prompt.json"
        if json_file.exists():
            with open(json_file, "r") as f:
                json_data = json.load(f)
                if json_data.get("created_with") == "pix2pix":
                    created_with_pix2pix = True
                else:
                    created_with_pix2pix = False
        else:
            print(f"prompt.json file does not exist for {row['img_id']}")
            errors.append(row['img_id'])
            continue
    
        
        # now write everything back to the json file
        json_item["created_with"] = "pix2pix" if created_with_pix2pix else "informative-drawings"
        new_json_file = target_dir / f"{row['img_id']}" / "metadata.json"
        with open(new_json_file, "w") as f:
            json.dump(json_item, f)
        
    # write errors to a file
    with open(f"errors/errors_{file_number}.json", "w") as f:
        json.dump(errors, f)
            

def repopulate_all_json_files():
    # parallelize this by processing 24 files at a time in parallel
    # use an executor to run the function in parallel
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
        futures = [executor.submit(repopulate_json_files, i) for i in range(24)]
        print(f"Submitted {len(futures)} futures")
        concurrent.futures.wait(futures)
    
    print("Finished processing all files")


def copy_over_all_files():
    import shutil
    finalpix2pix_dir = Path("/bigdrive/datasets/sketchy2pix/final-pix2pix")
    final_dir = Path("/bigdrive/datasets/sketchy2pix/final")
    i = 0
    already_exists_files = []
    
    for dir in finalpix2pix_dir.iterdir():
        if dir.is_dir():
            target_dir = final_dir / dir.name
            if target_dir.exists():
                print(f"Skipping {dir} because it already exists in {final_dir}")
                already_exists_files.append(dir.name)
                continue
            shutil.copytree(dir, target_dir)
            i += 1
            if i % 1000 == 0:
                print(f"Copied {i} directories")
    
    print(f"Copied {i} directories")
    print(f"Already exists directories: {already_exists_files}")
    # save to errors/already_exists_files.json
    with open("errors/already_exists_files.json", "w") as f:
        json.dump(already_exists_files, f)

repopulate_all_json_files()

# copy_over_all_files()

# set_latest_file_processed(target_dir, -1)

# def test():
    # img_id_to_mask = get_memory_map_for_masks()
    # net_G = get_net_G()
    # pix2pix_model = get_pix2pix_model()
    
    # generate_subset_dataset_for_file(input_images_dir / "0.arrow", img_id_to_mask, net_G, pix2pix_model, is_informative_drawings=False)

    # input_image_data = read_arrow_file(input_images_dir / "0.arrow")
    
    # for idx, row in input_image_data.iterrows():
    #     if not row['Instruction_VLM-LLM']:
    #         pprint(row)
# test()
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# generate_entire_subset_dataset(i = 20, i_end = 24)