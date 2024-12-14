



####
#
#
# This code finds the all stained cores for each patient id and places them in a canvas,
#
# Just provide these two paths
#
# base_path = '/data_slow2/ve59kedo/TMA/TMA_Cores_for_Hamed/'
# base_save_path = '/data_slow2/ve59kedo/TMA/TMA_ordered_png'
#
###


import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def find_patient_images(base_path, patient_id):
    patient_images = {}
    
    expected_stainings = [
        "CD163", "CD3", "CD56", "CD68", "CD8", "HE", "MHC1", "PDL1"
    ]
    
    for staining in expected_stainings:
        patient_images[staining] = []

    for staining in expected_stainings:
        dir_path = os.path.join(base_path, f"tma_tumorcenter_{staining}")
        if os.path.isdir(dir_path):
            for file_name in os.listdir(dir_path):
                if file_name.endswith('.png') and f"patient{patient_id}" in file_name:
                    patient_images[staining].append(os.path.join(dir_path, file_name))

    total_images = sum(len(images) for images in patient_images.values())
    if total_images == 0:
        print(f"Expected images, but found {total_images} for patient {patient_id}")
    
    return patient_images


def create_composite_image(images_dict, save_path):

    if not images_dict or all(len(imgs) == 0 for imgs in images_dict.values()):
        print("The image dictiionary is empty.")
        return  

    width, height = 4500, 4500  # since each image is 500x500. so 500*9 staining
    num_stainings = len(images_dict)
    max_images = max(len(imgs) for imgs in images_dict.values())
    canvas = Image.new('RGB', (width * num_stainings, height * max_images), (255, 255, 255))
    
    for colm_index, (staining, file_paths) in enumerate(sorted(images_dict.items())):
        extended_file_paths = file_paths + [None] * (max_images - len(file_paths))
        
        for row_index, file_path in enumerate(extended_file_paths):
            if file_path is None:
                img = Image.new('RGB', (500, 500), (255, 255, 255))
            else:
                img = Image.open(file_path)
            

            position = (colm_index * width, row_index * height)
            canvas.paste(img, position)
    

    canvas.save(save_path, format='png', dpi=(800, 800))


# path where the directories are located
base_path = '/data_slow2/ve59kedo/TMA/TMA_Cores_for_Hamed/'
base_save_path = '/data_slow2/ve59kedo/TMA/TMA_ordered_png'
# iterating for all ideas and finding the core and placing it in the corresponding location

for id in range(1, 770):
    id = f"{id:03d}"
    print(f'processig ID: {id}')
    patient_images = find_patient_images(base_path, id)
    save_path = f"{base_save_path}/DSTMA_patient_{id}.tiff"  # DSTMA = diffrently stained TMA
    create_composite_image(patient_images, save_path)