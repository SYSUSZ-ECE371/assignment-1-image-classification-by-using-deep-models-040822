import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import random_split
import os
import time
import copy
from torchvision.models import ResNet18_Weights

# Set data directory 数据目录
data_dir = 'flower_dataset'

# Data augmentation and normalization for training and validation 数据增强和归一化
data_transforms = transforms.Compose([
        # GRADED FUNCTION: Add five data augmentation methods, Normalizating and Tranform to tensor
        # TODO：添加五种数据增强方法，归一化并把数据转换为张量
        # Ref： https://zhuanlan.zhihu.com/p/476220305
        # Ref：https://pytorch.ac.cn/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html
        ### START SOLUTION HERE ###
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转图像
            transforms.RandomVerticalFlip(p=0.5),    # 随机垂直翻转图像
            transforms.RandomRotation(degrees=30),   # 随机旋转，30度
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 改变图像颜色的对比度、饱和度和hue
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),  # 随机裁剪图像并缩放到224x224
            transforms.ToTensor(),  # 转化为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
        ])
        ### END SOLUTION HERE ###
])

# Load the entire dataset 载入整个数据集
full_dataset = datasets.ImageFolder(data_dir, data_transforms)

# Automatically split into 80% train and 20% validation 划分数据集为80%训练集和20%验证集
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Use DataLoader for both train and validation datasets 使用DataLoader加载训练集和验证集
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

# Get class names from the dataset 获取类别名称
class_names = full_dataset.classes

# Load pre-trained model and modify the last layer 载入预训练模型并修改最后一层
model = models.resnet18(pretrained=True)

#model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# GRADED FUNCTION: Modify the last fully connected layer of model
# TODO：修改模型的最后一个全连接层
### START SOLUTION HERE ###
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
### END SOLUTION HERE ###


# TODO：定义损失函数
# GRADED FUNCTION: Define the loss function
### START SOLUTION HERE ###
loss_fn = nn.CrossEntropyLoss() 
criterion = loss_fn
### END SOLUTION HERE ###

# TODO：定义优化器
# GRADED FUNCTION: Define the optimizer
### START SOLUTION HERE ###
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
### END SOLUTION HERE ###

# Learning rate scheduler 学习率调度器，StepLR为线性退火
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Print learning rate for current epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Learning Rate: {current_lr:.6f}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        
                        # GRADED FUNCTION: Backward pass and optimization
                        # TODO：反向传播和优化
                        ### START SOLUTION HERE ###
                        loss.backward()
                        optimizer.step()
                        ### END SOLUTION HERE ###

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()  # Update learning rate based on scheduler

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

            # Save the model if validation accuracy is the best so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save the best model
                save_dir = 'Ex2/work_dir'
                os.makedirs(save_dir, exist_ok=True)

               # GRADED FUNCTION: Save the best model
               # TODO：保存最好的模型
                ### START SOLUTION HERE ###
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                ### END SOLUTION HERE ###

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    curve_print(train_loss_history, train_acc_history, val_loss_history, val_acc_history) #画训练曲线
    return model

def curve_print(train_loss_history, train_acc_history, val_loss_history, val_acc_history):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))

    # Plot train and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot train and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.suptitle('Training and Validation Curves', fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.savefig('training_validation_curves.png')

# Train the model
if __name__ == "__main__":
    # Check if GPU is available and use it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=25)