!pip install laplace-torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from laplace import Laplace
import os, zipfile
from google.colab import files

# 随机种子
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 换成了 MNIST 数据集，防止跟别人重样
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

val_size = 5000
train_size = len(train_dataset) - val_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 2. 模型，Dropout 改成了 0.3
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64*14*14, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3) # 改成了0.3
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN().to(device)

# 3. 训练，Epoch改成了8，lr改成了0.005
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005) # 改成了0.005
epochs = 8 # 改成了8
train_losses = []
val_accs = []

print("开始训练...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = correct / total
    val_accs.append(val_acc)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}')

plt.figure()
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('training_curve.png')
plt.close()

# 评估 MAP 模型
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    acc = correct / total
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return acc, all_probs, all_labels

map_acc, map_probs, map_labels = evaluate_model(model, test_loader)
print(f'MAP Accuracy: {map_acc:.4f}')

# 拉普拉斯近似
print("开始拉普拉斯近似...")
la = Laplace(model, 'classification',
             subset_of_weights='last_layer',
             hessian_structure='diag')
la.fit(train_loader)
la.optimize_prior_precision(method='gridsearch', val_loader=val_loader,
                            pred_type='glm', link_approx='probit')

def laplace_predict(la, loader):
    all_probs = []
    all_labels = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        probs = la(images, pred_type='glm', link_approx='probit')
        all_probs.append(probs.cpu().detach().numpy())
        all_labels.append(labels.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_probs, all_labels

la_probs, la_labels = laplace_predict(la, test_loader)
la_preds = np.argmax(la_probs, axis=1)
la_acc = np.mean(la_preds == la_labels)
print(f'Laplace Accuracy: {la_acc:.4f}')

# 指标计算
def nll(probs, labels):
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.mean(np.log(probs[np.arange(len(labels)), labels]))

def ece(probs, labels, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidences = np.max(probs, axis=1)
    predicted_labels = np.argmax(probs, axis=1)
    accuracies = predicted_labels == labels
    ece_val = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece_val += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece_val

def entropy(probs):
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)

map_nll = nll(map_probs, map_labels)
la_nll = nll(la_probs, la_labels)
map_ece = ece(map_probs, map_labels)
la_ece = ece(la_probs, la_labels)
map_entropy = entropy(map_probs).mean()
la_entropy = entropy(la_probs).mean()

print(f'MAP NLL: {map_nll:.4f}, Laplace NLL: {la_nll:.4f}')
print(f'MAP ECE: {map_ece:.4f}, Laplace ECE: {la_ece:.4f}')
print(f'MAP Entropy: {map_entropy:.4f}, Laplace Entropy: {la_entropy:.4f}')

# 混淆矩阵
cm = confusion_matrix(la_labels, la_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.close()

# 不确定性直方图
plt.figure()
plt.hist(map_entropy, bins=50, alpha=0.5, label='MAP')
plt.hist(la_entropy, bins=50, alpha=0.5, label='Laplace')
plt.xlabel('Predictive Entropy')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('uncertainty_hist.png')
plt.close()

# 保存结果 CSV
results = pd.DataFrame({
    'Method': ['MAP', 'Laplace'],
    'Accuracy': [map_acc, la_acc],
    'NLL': [map_nll, la_nll],
    'ECE': [map_ece, la_ece],
    'Entropy': [map_entropy, la_entropy]
})
results.to_csv('results.csv', index=False)
print(results)

# ===== 自动打包下载 =====
print("正在打包文件...")
files_to_zip = ['training_curve.png', 'confusion_matrix.png', 'uncertainty_hist.png', 'results.csv']
zip_name = 'Assignment_Results.zip'

with zipfile.ZipFile(zip_name, 'w') as zipf:
    for f in files_to_zip:
        if os.path.exists(f):
            zipf.write(f)
            print(f"已加入: {f}")
        else:
            print(f"警告：找不到 {f}")

files.download(zip_name)
print("✅ 打包完成！请查看你电脑的浏览器下载提示，文件已下载。")
