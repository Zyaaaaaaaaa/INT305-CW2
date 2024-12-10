import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 1. 深度模型
class DeepMNIST(nn.Module):
    def __init__(self):
        super(DeepMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 2. BatchNorm模型
class BatchNormMNIST(nn.Module):
    def __init__(self):
        super(BatchNormMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 3. 宽度模型
class WideMNIST(nn.Module):
    def __init__(self):
        super(WideMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 128 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class ModelTrainer:
    def __init__(self, model, model_name, config, model_dir='models'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model_name = model_name
        self.config = config
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # 配置优化器
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        else:
            self.optimizer = optim.SGD(model.parameters(), 
                                     lr=config['learning_rate'],
                                     momentum=config.get('momentum', 0.9))

        # 准备数据
        self.train_loader, self.test_loader = self.load_data()
        
        # 记录训练历史
        self.train_losses = []
        self.test_accuracies = []

    def load_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('data', train=False, transform=transform)
        
        train_loader = DataLoader(train_dataset, 
                                batch_size=self.config['batch_size'], 
                                shuffle=True)
        test_loader = DataLoader(test_dataset, 
                               batch_size=1000, 
                               shuffle=False)
        
        return train_loader, test_loader

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'{self.model_name} - Train Epoch: {epoch} '
                      f'[{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)]\t'
                      f'Loss: {loss.item():.6f}')
        
        return epoch_loss / len(self.train_loader)

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        
        print(f'\n{self.model_name} - Test set: Average loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(self.test_loader.dataset)} '
              f'({accuracy:.2f}%)\n')
        
        return accuracy

    def train(self, epochs):
        best_accuracy = 0
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            accuracy = self.test()
            self.test_accuracies.append(accuracy)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                model_path = os.path.join(self.model_dir, 
                                        f'mnist_{self.model_name}_{accuracy:.2f}.pt')
                torch.save(self.model.state_dict(), model_path)
                print(f'Saved {self.model_name} model to {model_path}')
        
        return best_accuracy, self.train_losses, self.test_accuracies

def plot_comparison(results, save_dir='models'):
    plt.figure(figsize=(15, 5))
    
    # Plot training losses
    plt.subplot(121)
    for model_name, (_, losses, _) in results.items():
        plt.plot(losses, label=model_name)
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot test accuracies
    plt.subplot(122)
    for model_name, (_, _, accuracies) in results.items():
        plt.plot(accuracies, label=model_name)
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'))
    plt.close()

def main():
    # 模型配置
    configs = {
        'deep': {
            'batch_size': 64,
            'learning_rate': 0.01,
            'optimizer': 'adam',
            'epochs': 10
        },
        'batchnorm': {
            'batch_size': 128,
            'learning_rate': 0.1,
            'optimizer': 'sgd',
            'momentum': 0.9,
            'epochs': 10
        },
        'wide': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'epochs': 10
        }
    }
    
    # 训练所有模型
    results = {}
    models = {
        'deep': DeepMNIST(),
        'batchnorm': BatchNormMNIST(),
        'wide': WideMNIST()
    }
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name} model...")
        trainer = ModelTrainer(model, model_name, configs[model_name])
        best_acc, losses, accuracies = trainer.train(configs[model_name]['epochs'])
        results[model_name] = (best_acc, losses, accuracies)
        print(f"{model_name} model best accuracy: {best_acc:.2f}%")
    
    # 绘制比较图
    plot_comparison(results)
    
    # 打印最终结果
    print("\nFinal Results:")
    for model_name, (best_acc, _, _) in results.items():
        print(f"{model_name} model: {best_acc:.2f}%")

if __name__ == '__main__':
    main()