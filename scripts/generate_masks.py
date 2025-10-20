import torch
import cv2
import numpy as np
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tqdm import tqdm

# Cargar SAM
sam = sam_model_registry["vit_h"](checkpoint="C:/train_bg_removal/sam/sam_vit_h_4b8939.pth")
sam.to('cuda')
mask_generator = SamAutomaticMaskGenerator(sam)

# Procesar imágenes
img_dir = Path("C:/train_bg_removal/dataset/images")
mask_dir = Path("C:/train_bg_removal/dataset/masks")

for img_path in tqdm(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))):
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Generar máscaras
    masks = mask_generator.generate(img_rgb)
    
    # Combinar TODAS las máscaras en una sola
    if masks:
        combined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for m in masks:
            combined_mask = np.maximum(combined_mask, (m['segmentation'] * 255).astype(np.uint8))
        
        # Guardar
        mask_name = img_path.stem + '.png'
        mask_path = mask_dir / mask_name
        cv2.imwrite(str(mask_path), combined_mask)

print("Máscaras generadas!")
