from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score

from GAT import *
from adj_matrix_generate import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% 读取数据构建图
df = pd.read_csv('faults.csv')
features = df.iloc[:, :-7].values
labels = df.iloc[:, -7:].values
labels = np.argmax(labels, axis=1)
num_classes = len(np.unique(labels))

scaler = StandardScaler()
features = scaler.fit_transform(features)

# 设置阈值
train_ratio = 0.8

#%%  使用SMOTE对数据集进行上采样
sm = SMOTE(random_state=42)
features_resampled, labels_resampled = sm.fit_resample(features, labels)

# 使用generate_masks函数
train_mask, test_mask = generate_masks(features_resampled, labels_resampled, train_ratio)

# Graph Construction: changed the hyperparm 'method' to select different methods
g = build_dgl_graph(features_resampled, labels_resampled, method='fuse', param=[]).to(device)

# 将掩码添加到图的ndata中
g.train_mask = train_mask.to(device)
g.test_mask = test_mask.to(device)


#%% 初始化GAT模型
def train(g, model, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc_test = 0
    best_epoch = 0
    best_metrics = None

    for epoch in range(epochs):
        model.train()
        logits = model(g, g.ndata['feat'])
        loss = loss_fn(logits[g.train_mask], g.ndata['label'][g.train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_train, prec_train, recall_train, f1_train = evaluate(g, model, g.train_mask)
        acc_test, prec_test, recall_test, f1_test = evaluate(g, model, g.test_mask)

        if acc_test > best_acc_test:
            best_acc_test = acc_test
            best_epoch = epoch
            best_metrics = (prec_test, recall_test, f1_test)

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.3f},Train Accuracy: {acc_train:.3f}, Test Accuracy: {acc_test:.3f},"
                  f"Test Precision: {prec_test:.3f},Test Recall: {recall_test:.3f},Test F1: {f1_test:.3f}")

    best_prec, best_recall, best_f1 = best_metrics
    print(f"\nBest Test Accuracy at Epoch {best_epoch+1}: Accuracy: {best_acc_test:.3f}, Precision: {best_prec:.3f}, Recall: {best_recall:.3f}, F1: {best_f1:.3f}")

def evaluate(g, model, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, g.ndata['feat'])
        logits = logits[mask]
        labels = g.ndata['label'][mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        acc = correct.item() * 1.0 / len(labels)
        precision = precision_score(labels.cpu(), indices.cpu(), average='macro', zero_division=0)
        recall = recall_score(labels.cpu(), indices.cpu(), average='macro', zero_division=0)
        f1 = f1_score(labels.cpu(), indices.cpu(), average='macro', zero_division=0)
    return acc, precision,recall ,f1

#%%
in_dim = features.shape[1]
hidden_dim = 80
num_heads = 15
epochs = 1000
lr = 0.005
dropout = 0.1

model = GAT(in_dim, hidden_dim,  num_heads=num_heads, out_dim=num_classes, dropout=dropout).to(device)

train(g, model, epochs, lr)


