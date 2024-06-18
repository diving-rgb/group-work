import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 读取数据
train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

# 处理数据
# 训练集时间特征提取
train_data['Dates'] = pd.to_datetime(train_data['Dates'])
train_data['Year'] = train_data['Dates'].dt.year
train_data['Month'] = train_data['Dates'].dt.month
train_data['Day'] = train_data['Dates'].dt.day
train_data['Hour'] = train_data['Dates'].dt.hour
train_data['DayOfWeek'] = train_data['Dates'].dt.dayofweek

# 测试集时间特征提取
test_data['Dates'] = pd.to_datetime(test_data['Dates'])
test_data['Year'] = test_data['Dates'].dt.year
test_data['Month'] = test_data['Dates'].dt.month
test_data['Day'] = test_data['Dates'].dt.day
test_data['Hour'] = test_data['Dates'].dt.hour
test_data['DayOfWeek'] = test_data['Dates'].dt.dayofweek

# 类别数据映射
le = LabelEncoder()
train_data['Category'] = le.fit_transform(train_data['Category'])

# 选择特征
geo_features = ['X', 'Y']
time_features = ['Year', 'Month', 'Day', 'Hour', 'DayOfWeek']
X_train_geo = train_data[geo_features].values
X_train_time = train_data[time_features].values
y_train = train_data['Category'].values

scaler_geo = StandardScaler()
scaler_time = StandardScaler()

# 训练集标准化特征
X_train_geo = scaler_geo.fit_transform(X_train_geo)
X_train_time = scaler_time.fit_transform(X_train_time)

# 测试集标准化特征
X_test_geo = test_data[geo_features].values
X_test_time = test_data[time_features].values
X_test_geo = scaler_geo.transform(X_test_geo)  # 使用相同的scaler进行测试集的标准化
X_test_time = scaler_time.transform(X_test_time)

# 数据集
class CrimeDataset(Dataset):
    def __init__(self, X_geo, X_time, y=None):
        self.X_geo = torch.tensor(X_geo, dtype=torch.float32)
        self.X_time = torch.tensor(X_time, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X_geo)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_geo[idx], self.X_time[idx], self.y[idx]
        else:
            return self.X_geo[idx], self.X_time[idx]

# 创建训练集和测试集的Dataset和DataLoader
train_dataset = CrimeDataset(X_train_geo, X_train_time, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = CrimeDataset(X_test_geo, X_test_time)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义CNN+LSTM模型
class CNNLSTM(nn.Module):
    def __init__(self, input_size_geo, input_size_time, hidden_size, num_layers, num_classes):
        super(CNNLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.cnn_output_size = self._get_cnn_output_size(input_size_geo)
        self.lstm = nn.LSTM(input_size_time, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.cnn_output_size + hidden_size * 2, num_classes)

    def _get_cnn_output_size(self, input_size_geo):
        with torch.no_grad():
            x = torch.zeros(1, 1, input_size_geo)
            x = self.cnn(x)
            return x.numel()

    def forward(self, x_geo, x_time):
        batch_size = x_geo.size(0)

        # CNN for geo features
        x_geo = x_geo.view(batch_size, 1, -1)
        x_geo = self.cnn(x_geo)
        x_geo = x_geo.view(batch_size, -1)

        # LSTM for 时间
        h0 = torch.zeros(self.num_layers * 2, x_time.size(0), self.hidden_size).to(x_time.device)
        c0 = torch.zeros(self.num_layers * 2, x_time.size(0), self.hidden_size).to(x_time.device)
        x_time, _ = self.lstm(x_time.unsqueeze(1), (h0, c0))
        x_time = x_time[:, -1, :]

        # Combine features
        x = torch.cat((x_geo, x_time), dim=1)
        out = self.fc(x)
        return out

# 定义模型参数
input_size_geo = X_train_geo.shape[1]
input_size_time = X_train_time.shape[1]
hidden_size = 128
num_layers = 2
num_classes = len(le.classes_)

# 初始化模型
model = CNNLSTM(input_size_geo, input_size_time, hidden_size, num_layers, num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 180
print("epoch start:")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for geo_inputs, time_inputs, labels in train_loader:
        geo_inputs, time_inputs, labels = geo_inputs.to(device), time_inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(geo_inputs, time_inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
print("start")
# 在测试集上预测并生成提交结果
model.eval()
predictions = []
with torch.no_grad():
    for geo_inputs, time_inputs in test_loader:
        geo_inputs, time_inputs = geo_inputs.to(device), time_inputs.to(device)
        outputs = model(geo_inputs, time_inputs)
        probabilities = torch.softmax(outputs, dim=1)
        predictions.extend(probabilities.cpu().numpy())

# 生成提交结果的 DataFrame
columns = ['Id'] + list(le.inverse_transform(range(num_classes)))
submission_df = pd.DataFrame(columns=columns)
submission_df['Id'] = test_data['Id']
for i, class_name in enumerate(le.inverse_transform(range(num_classes))):
    submission_df[class_name] = [round(max(min(pred[i], 1 - 10**-15), 10**-15), 15) for pred in predictions]

# 将结果保存为 CSV 文件
submission_df.to_csv('./data/crime_predictions.csv', index=False)

print("end")