import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# 하이퍼파라미터 설정
batch_size = 64
learning_rate = 0.001
epochs = 2

# MNIST 데이터셋 다운로드 및 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 데이터 로더 생성
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델 정의
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = Net()

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 이미지 저장 함수
def save_image(img, filename):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    img_transposed = np.transpose(npimg, (1, 2, 0))
    img = Image.fromarray((img_transposed * 255).astype(np.uint8)) # PIL Image로 변환
    img.save(filename)

# 이미지 저장 폴더 생성
output_dir = 'mnist_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 클래스 이름
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# 학습 루프
for epoch in range(epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        # 순전파
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 역전파 및 가중치 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, i + 1, len(train_loader), loss.item()))

            # 예측 결과 시각화 (일부 이미지)
            dataiter = iter(test_loader)
            images, labels = next(dataiter)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # 이미지 저장
            filename = os.path.join(output_dir, f'epoch_{epoch+1}_step_{i+1}.png')
            save_image(torchvision.utils.make_grid(images), filename)
            print(f'이미지 저장: {filename}')

            print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
            print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# 모델 평가 (수정 없음)
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
