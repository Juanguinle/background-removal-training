import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A

# Configuración
IMG_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 0.00005
DEVICE = 'cuda'

# Dataset personalizado
class PersonDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.images = sorted(list(self.img_dir.glob('*.jpg')) + list(self.img_dir.glob('*.png')))
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask_path = self.mask_dir / (img_path.stem + '.png')
        mask = cv2.imread(str(mask_path), 0)
        
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)
        
        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return img, mask

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(p=0.2),
])

if __name__ == '__main__':
    dataset = PersonDataset(
        'C:/train_bg_removal/dataset/images',
        'C:/train_bg_removal/dataset/masks',
        transform=train_transform
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = smp.DeepLabV3Plus(
        encoder_name='mobilenet_v2',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation=None
    )
    model.to(DEVICE)

    criterion = smp.losses.DiceLoss(mode='binary')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'C:/train_bg_removal/models/best_model.pth')
            print('Modelo guardado!')

    print('Entrenamiento completo!')
