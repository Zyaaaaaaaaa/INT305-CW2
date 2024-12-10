import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from datetime import datetime

class SimpleBaseline(nn.Module):
    def __init__(self):
        super(SimpleBaseline, self).__init__()
        # 简化的网络结构
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)  # 单个卷积层
        self.fc1 = nn.Linear(16 * 13 * 13, 32)  # 减少神经元数量
        self.fc2 = nn.Linear(32, 10)
        self.dropout = nn.Dropout(0.5)  # 降低dropout率

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class MNISTTrainer:
    def __init__(self, model_dir='models', data_dir='data'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 创建保存目录
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # 训练配置
        self.batch_size = 64
        self.learning_rate = 0.1
        
        # 加载数据
        self.train_loader, self.test_loader = self.load_data(data_dir)
        
        # 初始化模型
        self.model = SimpleBaseline().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), 
                                 lr=self.learning_rate, 
                                 momentum=0.9)
        
        # 训练历史
        self.train_losses = []
        self.test_accuracies = []
        self.misclassified_examples = []

    def load_data(self, data_dir):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        
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
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        return epoch_loss / len(self.train_loader)

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        self.misclassified_examples = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
                # 存储错误分类的例子
                mask = ~pred.eq(target.view_as(pred)).squeeze()
                misclassified_data = data[mask]
                misclassified_pred = pred[mask]
                misclassified_target = target[mask]
                misclassified_conf = torch.exp(output[mask]).max(dim=1)[0]
                
                for i in range(len(misclassified_data)):
                    if len(self.misclassified_examples) < 10:
                        self.misclassified_examples.append({
                            'image': misclassified_data[i].cpu(),
                            'predicted': misclassified_pred[i].item(),
                            'actual': misclassified_target[i].item(),
                            'confidence': misclassified_conf[i].item()
                        })
        
        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        
        print(f'\nTest set: Average loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(self.test_loader.dataset)} ({accuracy:.2f}%)\n')
        
        return accuracy

    def save_results(self, accuracy):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(self.model_dir, f'mnist_simple_{accuracy:.2f}_{timestamp}.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'train_losses': self.train_losses,
            'test_accuracies': self.test_accuracies
        }, model_path)
        print(f'Saved model to {model_path}')

    def plot_results(self):
        plt.figure(figsize=(12, 4))
        
        # 绘制训练损失
        plt.subplot(121)
        plt.plot(self.train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        # 绘制测试准确率
        plt.subplot(122)
        plt.plot(self.test_accuracies)
        plt.title('Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(os.path.join(self.model_dir, f'training_results_{timestamp}.png'))
        plt.close()

    def plot_misclassified(self):
        plt.figure(figsize=(15, 6))
        for i, example in enumerate(self.misclassified_examples[:10]):
            plt.subplot(2, 5, i + 1)
            plt.imshow(example['image'].squeeze(), cmap='gray')
            plt.title(f'Pred: {example["predicted"]}\nTrue: {example["actual"]}\n'
                     f'Conf: {example["confidence"]:.2f}')
            plt.axis('off')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, f'misclassified_{timestamp}.png'))
        plt.close()

    def train(self, epochs=10):
        best_accuracy = 0
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            accuracy = self.test()
            self.test_accuracies.append(accuracy)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save_results(accuracy)
        
        self.plot_results()
        self.plot_misclassified()
        return best_accuracy

if __name__ == '__main__':
    trainer = MNISTTrainer()
    best_accuracy = trainer.train(epochs=10)
    print(f'Training completed with best accuracy: {best_accuracy:.2f}%')