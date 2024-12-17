from datasets import load_dataset
from torchvision import transforms
import torch
from tqdm import tqdm
from PIL import Image as PILImage
from datasets import Dataset, DatasetDict, Image
import json
import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

def load_image_as_bytes(image_path):
    with open(image_path, "rb") as f:
        return f.read()

def process_and_save_to_parquet(source_folder, output_parquet, batch_size):
    batch_data = {
        'image': [],
        'conditioning_image': [],
        'text': [],
        'sketch': [],
        'mask': [],
        'img_id': [],
        'target_img_dataset': [],
        'Instruction_Ref_Dataset': [],
        'ann_id': [],
        'created_with': [],
        'json_contents': []
    }
    batch_counter = 0
    parquet_writer = None

    folder_list = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]
    for folder_name in tqdm(folder_list, desc="Processing Folders"):
        folder_path = os.path.join(source_folder, folder_name)

        if os.path.isdir(folder_path):
            src_image, tgt_image, mask, sketch = None, None, None, None
            edit_dict = None

            # Iterate through files in the folder
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                if file_name == 'metadata.json':
                    with open(file_path, 'r') as f:
                        data = f.read()
                        if data == None:
                            continue
                        edit_dict = json.loads(data)
                        
                elif file_name.endswith('.png'):
                    file_number = int(file_name.split('_')[1][0])
                    image_data = load_image_as_bytes(file_path)
                    if file_number == 0:
                        src_image = image_data
                    elif file_number == 1:
                        tgt_image = image_data
                    elif file_number == 2:
                        mask = image_data
                    elif file_number == 3:
                        sketch = image_data
            
            # Skip folder if required files are missing
            edit_prompt = edit_dict.get('edit', None)
            checks = [src_image, tgt_image, mask, sketch, edit_prompt, edit_dict]
            
            if not all(checks):
                error = ""
                for check in checks:
                    if check is None:
                        error += f"{check} is missing. "
                if edit_prompt == "":
                    error += f"edit_prompt is missing. "

                with open("logs.json", 'a') as f:
                    f.write(f"Skipping folder {folder_name} due to missing files. {error}\n")
                continue

            # Add the data to the batch
            batch_data['image'].append(tgt_image)
            batch_data['conditioning_image'].append(src_image)
            batch_data['text'].append(edit_prompt)
            batch_data['sketch'].append(sketch) 
            batch_data['mask'].append(mask)
            json_arr = ['img_id', 'target_img_dataset', 'Instruction_Ref_Dataset', 'ann_id', 'created_with']
            for label in json_arr:
                batch_data[label].append(edit_dict.get(label, None))
            
            batch_data['json_contents'].append(edit_dict)

            # Save batch to Parquet when batch size is reached
            if len(batch_data['image']) >= batch_size:
                df = pd.DataFrame(batch_data)
                table = pa.Table.from_pandas(df)

                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(output_parquet, table.schema)

                parquet_writer.write_table(table)
                batch_data = {k: [] for k in batch_data}  # Reset batch data
                batch_counter += 1

    # Save any remaining data
    if len(batch_data['image']) > 0:
        df = pd.DataFrame(batch_data)
        table = pa.Table.from_pandas(df)
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(output_parquet, table.schema)
        parquet_writer.write_table(table)

    # Close the Parquet writer
    if parquet_writer:
        parquet_writer.close()

    print(f"Dataset saved to {output_parquet}.")

def upload_to_huggingface(output_parquet, hugging_face):
    dataset = Dataset.from_parquet(output_parquet)
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("conditioning_image", Image())
    dataset = dataset.cast_column("sketch", Image())
    dataset = dataset.cast_column("mask", Image())
    dataset.push_to_hub(hugging_face, private=False)
    print("Uploaded to huggingface")

source_folder = "/bigdrive/datasets/sketchy2pix/final"
output_parquet = "sketchy2pix_dataset.parquet"
hugging_face = "genecodes/sketchybusiness"

process_and_save_to_parquet(source_folder, output_parquet, batch_size=10000)
upload_to_huggingface(output_parquet, hugging_face)
