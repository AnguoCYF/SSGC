import pandas as pd
from imblearn.over_sampling import SMOTE

from adj_matrix_generate import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% 读取数据
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

# 构建DGL图
g = build_dgl_graph(features_resampled, labels_resampled, method='fuse', param=15).to(device)

# 将掩码添加到图的ndata中
g.train_mask = train_mask.to(device)
g.test_mask = test_mask.to(device)

#%%

# AF
X_train = features_resampled[g.train_mask.cpu().numpy()]
X_test = features_resampled[g.test_mask.cpu().numpy()]

y_train = labels_resampled[g.train_mask.cpu().numpy()]
y_test = labels_resampled[g.test_mask.cpu().numpy()]

#%%
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Use Support Vector Machine (SVM) for classification
model_svm = SVC(random_state=42).fit(X_train, y_train)
y_preds_svm = model_svm.predict(X_test)

prec_svm, rec_svm, f1_svm, num_svm = precision_recall_fscore_support(y_test, y_preds_svm, average='weighted')
acc_svm = accuracy_score(y_test, y_preds_svm)

print("SVM Classifier")
print("Weighted Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f \nAccuracy:%.3f" % (prec_svm, rec_svm, f1_svm, acc_svm))

#%%
# Use Logistic Regression for classification
model_lr = LogisticRegression(max_iter=500, random_state=42).fit(X_train, y_train)
y_preds_lr = model_lr.predict(X_test)

prec_lr, rec_lr, f1_lr, num_lr = precision_recall_fscore_support(y_test, y_preds_lr, average='weighted')
acc_lr = accuracy_score(y_test, y_preds_lr)

print("Logistic Regression Classifier")
print("Weighted Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f \nAccuracy:%.3f" % (prec_lr, rec_lr, f1_lr, acc_lr))

#%%
from sklearn.neural_network import MLPClassifier

out_dim = X_train.shape[1]
num_classes = len(set(y_train))

# Train the MLP Classifier
model_mlp = MLPClassifier(hidden_layer_sizes=(out_dim,), activation='relu', max_iter=500, random_state=42)
model_mlp.fit(X_train, y_train)

# Make predictions
y_preds_mlp = model_mlp.predict(X_test)

# Evaluate the MLP Classifier
prec_mlp, rec_mlp, f1_mlp, num_mlp = precision_recall_fscore_support(y_test, y_preds_mlp, average='weighted')
acc_mlp = accuracy_score(y_test, y_preds_mlp)

print("MLP Classifier")
print("Weighted Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f \nAccuracy:%.3f" % (prec_mlp, rec_mlp, f1_mlp, acc_mlp))

#%%
from sklearn.tree import DecisionTreeClassifier

# Use Decision Tree for classification
model_dt = DecisionTreeClassifier().fit(X_train, y_train)
y_preds_dt = model_dt.predict(X_test)

prec_dt, rec_dt, f1_dt, num_dt = precision_recall_fscore_support(y_test, y_preds_dt, average='weighted')
acc_dt = accuracy_score(y_test, y_preds_dt)

print("Decision Tree Classifier")
print("Weighted Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f \nAccuracy:%.3f" % (prec_dt, rec_dt, f1_dt, acc_dt))
