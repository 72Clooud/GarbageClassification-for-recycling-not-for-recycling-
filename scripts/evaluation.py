import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from typing import List

from train import ModelHandler, CustomDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Config:
    
    def __init__(self):
        self.test_dir = './test_dir'
        self.model_path = './models/best_model.pth'
        self.categories = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
               'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']
        self.num_classes = len(self.categories)


class ModelTester:
    
    def __init__(self, model_path: str, num_classes: int, test_dir: str, categories: List[str]):
        self.model_handler = ModelHandler(num_classes, device=device)
        self.model_handler.load_model(model_path)
        self.model_handler.model.eval()
        self.test_dir = test_dir
        self.categories = categories
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def create_test_loader(self, batch_size=32):
        dataset = CustomDataset(self.test_dir, config.categories, self.transform)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return test_loader
    
    def test_model(self, test_loader):
        total_loss, correct = 0.0, 0
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model_handler.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                
        avg_loss = total_loss / len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)
        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy
    
if __name__ == "__main__":
    
    config = Config()
    
    model_tester = ModelTester(config.model_path, config.num_classes, config.test_dir, config.categories)
    test_loader = model_tester.create_test_loader()
    
    model_tester.test_model(test_loader)
    