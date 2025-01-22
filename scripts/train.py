import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from PIL import Image
from typing import List
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    
    def __init__(self, data_dir: str, categories: List[str], transform = None):
        self.data = []
        self.labels = []
        self.transform = transform

        for idx, category in enumerate(categories):
            category_path = os.path.join(data_dir, category)
            for file_name in tqdm(os.listdir(category_path), desc=f"Loading category: {category}", unit="file"):
                file_path = os.path.join(category_path, file_name)
                self.data.append(file_path)
                self.labels.append(idx)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        img_path = self.data[index]
        label = self.labels[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label



class DataProcessor:
    
    def __init__(self, data_dir: str, train_dir: str, test_dir: str, val_dir: str, categories: List[str]):
        self.data_dir = data_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir
        self.categories = categories
        
    def split_data(self, test_size=0.2, val_size=0.2) -> None:
        for category in self.categories:
            category_path = os.path.join(self.data_dir, category)
            images = os.listdir(category_path)
            
            train_val_images, test_images = train_test_split(images, test_size=test_size)
            train_images, val_images = train_test_split(train_val_images, test_size=val_size / (1 - test_size))
            
            os.makedirs(os.path.join(self.train_dir, category), exist_ok=True)
            os.makedirs(os.path.join(self.test_dir, category), exist_ok=True)
            os.makedirs(os.path.join(self.val_dir, category), exist_ok=True)
            
            for image in tqdm(train_images, desc=f"Copying train images for {category}", unit="image"):
                shutil.copyfile(os.path.join(category_path, image), os.path.join(self.train_dir, category, image))
            for image in tqdm(test_images, desc=f"Copying test images for {category}", unit="image"):
                shutil.copyfile(os.path.join(category_path, image), os.path.join(self.test_dir, category, image))
            for image in tqdm(val_images, desc=f"Copying val images for {category}", unit="image"):
                shutil.copyfile(os.path.join(category_path, image), os.path.join(self.val_dir, category, image))
                
    def create_data_loader(self, batch_size=64, test_batch_size=32):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        train_dataset = CustomDataset(self.train_dir, self.categories, transform)
        test_dataset = CustomDataset(self.test_dir, self.categories, transform)
        val_dataset = CustomDataset(self.val_dir, self.categories, transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)
        
        return train_loader, test_loader, val_loader
    
class ModelHandler:
    
    def __init__(self, num_classes, device):
        self.device = device
        self.model = self._create_model(num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=5, factor=0.1)
    
    def _create_model(self, num_classes):
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.08),
            nn.Linear(64, num_classes)
        )
        
        for param in model.features.parameters():
            param.requires_grad = False
        
        return model.to(self.device)
    
    def train_one_epoch(self, dataloader):
        self.model.train()
        epoch_loss, correct = 0.0, 0
        
        for images, labels in tqdm(dataloader, desc="Training", unit="batch"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
        
        epoch_loss /= len(dataloader.dataset)
        accuracy = correct / len(dataloader.dataset)
        
        return epoch_loss, accuracy

    def validate(self, dataloader):
        self.model.eval()
        epoch_loss, correct = 0.0, 0
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Validation", unit="batch"):
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                loss = self.criterion(output, labels)
                
                epoch_loss += loss.item() * images.size(0)
                correct += (output.argmax(1) == labels).sum().item()
            
        epoch_loss /= len(dataloader.dataset)
        accuracy = correct / len(dataloader.dataset)
        
        return epoch_loss, accuracy
    
    def train_model(self, train_loader, val_loader, num_epochs):
        best_acc = 0.0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.scheduler.step(val_acc)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_model('/kaggle/working/best_model.pth')

        return history
    
    def plot_history(self, history):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.legend()
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Val Accuracy')
        plt.legend()
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        plt.show()
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        

if __name__ == '__main__':
    
    categories = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
                  'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

    data_processor = DataProcessor('./data', './trina_dir', './test_data', categories)
    data_processor.split_data()
    
    train_loader, test_loader = data_processor.create_data_loader()
    
    model_handler = ModelHandler(num_classes=len(categories))
    
    history = model_handler.train_model(train_loader, test_loader, 25)
    
    model_handler.plot_history(history)
