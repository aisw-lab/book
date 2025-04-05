import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분입니다.
torch.manual_seed(3)

# CUDA 사용 가능하면 CUDA 사용, 아니면 CPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 불러온 데이터를 적용합니다.
my_data = 'ThoraricSurgery.csv'
Data_set = np.loadtxt(my_data, delimiter=",")

# 환자의 기록과 수술 결과를 X와 Y로 구분하여 저장합니다.
X = Data_set[:,0:17]
Y = Data_set[:,17]

# NumPy 배열을 PyTorch 텐서로 변환하고 장치(device)로 이동합니다.
X = torch.tensor(X, dtype=torch.float32).to(device)
Y = torch.tensor(Y, dtype=torch.float32).to(device)

# 딥러닝 모델 정의
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(17, 30)  # 입력층(17) -> 은닉층(30)
        self.fc2 = nn.Linear(30, 1)   # 은닉층(30) -> 출력층(1)
        self.relu = nn.ReLU()         # ReLU 활성화 함수
        self.sigmoid = nn.Sigmoid()   # Sigmoid 활성화 함수

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# 모델 인스턴스 생성 및 장치(device)로 이동
model = Net().to(device)

# 손실 함수 및 최적화 알고리즘 정의
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters())

# 딥러닝 학습
epochs = 100
batch_size = 10

for epoch in range(epochs):
    for i in range(0, len(X), batch_size):
        # 미니배치 생성
        X_batch = X[i:i+batch_size]
        Y_batch = Y[i:i+batch_size]

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), Y_batch) # squeeze()를 사용하여 차원 축소

        # Backward and optimize
        optimizer.zero_grad() # gradient 초기화
        loss.backward()       # backward pass
        optimizer.step()      # weight 업데이트

    # Epoch마다 Loss 출력 (선택 사항)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# (선택 사항) 학습된 모델 저장
# torch.save(model.state_dict(), 'thoracic_surgery_model.pth')

# (선택 사항) 모델 평가 (테스트 데이터가 있는 경우)
# model.eval() # 평가 모드로 설정
# with torch.no_grad(): # gradient 계산 X
#     test_outputs = model(X_test)
#     predicted = (test_outputs > 0.5).float()
#     accuracy = (predicted == Y_test).sum().item() / Y_test.size(0)
#     print(f'Accuracy: {accuracy:.4f}')
