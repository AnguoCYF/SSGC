import dgl
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
from scipy.sparse.csgraph import minimum_spanning_tree

def normalize_matrix(matrix):
    min_value = np.nanmin(matrix)
    max_value = np.nanmax(matrix)
    normalized_matrix = (matrix - min_value) / (max_value - min_value)
    normalized_matrix = np.nan_to_num(normalized_matrix)
    return normalized_matrix


def generate_mahdistance_threshold_adjacency_matrix(features, quantile=25):
    # 标准化特征
    # scaler = StandardScaler()
    # features = scaler.fit_transform(features)

    # 计算马氏距离矩阵
    cov_estimator = EmpiricalCovariance()
    cov_estimator.fit(features)
    inv_cov = np.linalg.inv(cov_estimator.covariance_)
    mahalanobis_distances = pdist(features, metric='mahalanobis', VI=inv_cov)
    distance_matrix = squareform(mahalanobis_distances)
    distance_matrix = normalize_matrix(distance_matrix)  # 归一化马氏距离矩阵

    # 计算阈值（距离矩阵的25分位数）
    threshold = np.percentile(distance_matrix, quantile)

    # 根据阈值创建邻接矩阵
    adjacency_matrix = (distance_matrix <= threshold).astype(float)
    np.fill_diagonal(adjacency_matrix, 0)

    # 转换为对称矩阵
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T
    adjacency_matrix[adjacency_matrix > 1] = 1  # 确保连接权重为 1

    return adjacency_matrix

def generate_euclidean_threshold_adjacency_matrix(features, quantile=25):
    # 标准化特征（根据需要决定是否使用）
    # scaler = StandardScaler()
    # features = scaler.fit_transform(features)

    # 计算欧氏距离矩阵
    euclidean_distances = pdist(features, metric='euclidean')
    distance_matrix = squareform(euclidean_distances)
    distance_matrix = normalize_matrix(distance_matrix)  # 归一化欧氏距离矩阵

    # 计算阈值（距离矩阵的25分位数）
    threshold = np.percentile(distance_matrix, quantile)

    # 根据阈值创建邻接矩阵
    adjacency_matrix = (distance_matrix <= threshold).astype(float)
    np.fill_diagonal(adjacency_matrix, 0)

    # 转换为对称矩阵
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T
    adjacency_matrix[adjacency_matrix > 1] = 1  # 确保连接权重为 1

    return adjacency_matrix


def generate_cosine_similarity_matrix(features, quantile=75):
    # scaler = StandardScaler()
    # features = scaler.fit_transform(features)
    similarity_matrix = cosine_similarity(features)

    # 计算阈值（距离矩阵的25分位数）
    threshold = np.percentile(similarity_matrix, quantile)

    # 根据阈值创建邻接矩阵
    adjacency_matrix = (similarity_matrix >= threshold).astype(float)
    np.fill_diagonal(adjacency_matrix, 0)

    # 转换为对称矩阵
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T
    adjacency_matrix[adjacency_matrix > 1] = 1  # 确保连接权重为 1

    return adjacency_matrix


def generate_knn_adjacency_matrix(features, k):
    # scaler = StandardScaler()
    # features = scaler.fit_transform(features)
    adjacency_matrix = kneighbors_graph(features, n_neighbors=k, mode='connectivity', include_self=False)
    adjacency_matrix = adjacency_matrix.toarray()

    # 将邻接矩阵转换为对称矩阵，使图变为无向图
    adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)

    return adjacency_matrix

def generate_mst_adjacency_matrix(features):
    # 计算欧氏距离矩阵
    euclidean_distances = pdist(features, metric='euclidean')
    distance_matrix = squareform(euclidean_distances)

    # 构建最小生成树
    mst = minimum_spanning_tree(distance_matrix)

    # 将最小生成树转换为邻接矩阵
    adjacency_matrix = mst.toarray()

    # 转换为对称矩阵
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T
    adjacency_matrix[adjacency_matrix > 1] = 1  # 确保连接权重为 1

    return adjacency_matrix


def build_dgl_graph(features, labels, method, param):
    if method == 'cosine':
        adjacency_matrix = generate_cosine_similarity_matrix(features, param)
    elif method == 'knn':
        adjacency_matrix = generate_knn_adjacency_matrix(features, param)
    elif method == 'mahalanobis':
        adjacency_matrix = generate_mahdistance_threshold_adjacency_matrix(features, param)
    elif method == 'euclidean':
        adjacency_matrix = generate_euclidean_threshold_adjacency_matrix(features, param)
    elif method == 'mst':
        adjacency_matrix = generate_mst_adjacency_matrix(features)
    elif method == 'fuse':
        adj_knn = generate_knn_adjacency_matrix(features, 1)
        adj_mst = generate_mst_adjacency_matrix(features)
        adj_cos = generate_cosine_similarity_matrix(features, 95)
        adj_mah = generate_mahdistance_threshold_adjacency_matrix(features, 5)
        adjacency_matrix = fuse_adjacency_matrices_similarity_based([adj_knn, adj_mst, adj_cos, adj_mah])
    else:
        raise ValueError("Invalid method. Choose from 'cosine', 'knn', or 'mahalanobis'.")

    edge_indices = np.where(adjacency_matrix != 0)
    edge_weights = torch.tensor(adjacency_matrix[edge_indices], dtype=torch.float32)
    src_nodes, dst_nodes = edge_indices

    dgl_graph = dgl.graph((src_nodes, dst_nodes), num_nodes=features.shape[0])
    dgl_graph = dgl.add_self_loop(dgl_graph)  # 添加自环

    # 添加自环的权重
    self_loop_weights = torch.ones(dgl_graph.number_of_nodes(), dtype=torch.float32)
    edge_weights = torch.cat([edge_weights, self_loop_weights], dim=0)

    dgl_graph.edata['w'] = edge_weights.view(-1, 1)
    dgl_graph.ndata['feat'] = torch.tensor(features, dtype=torch.float32)  # 添加节点特征
    dgl_graph.ndata['label'] = torch.tensor(labels, dtype=torch.long)  # 添加节点标签
    return dgl_graph


def get_edge_weight(dgl_graph, src, dst):
    # 查找连接源节点和目标节点的边ID
    edge_id = dgl_graph.edge_id(src, dst)

    # 根据边ID获取边权重
    edge_weight = dgl_graph.edata['w'][edge_id].item()
    return edge_weight


def generate_masks(features, labels, train_ratio, seed=38):
    num_nodes = len(labels)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # 使用train_test_split进行数据集划分
    _, _, _, _, train_idx, test_idx = train_test_split(features, labels, np.arange(num_nodes), train_size=train_ratio, random_state=seed)

    train_mask[train_idx] = True
    test_mask[test_idx] = True

    return train_mask, test_mask


def is_symmetric(matrix):
    # 获取矩阵的行数和列数
    rows = len(matrix)
    cols = len(matrix[0])
    # 如果行数和列数不相等，矩阵不是对称的
    if rows != cols:
        return False
    # 遍历矩阵的下三角部分，检查每个元素是否等于其对应的上三角部分的元素
    for i in range(rows):
        for j in range(i):
            if matrix[i][j] != matrix[j][i]:
                return False
    # 如果没有发现不相等的元素，矩阵是对称的
    return True


def similarity_measure(matrix1, matrix2):
    """
    计算矩阵之间的相似度，使用Frobenius范数计算矩阵之间的距离，然后将其转换为相似度。
    """
    distance = np.linalg.norm(matrix1 - matrix2, 'fro')
    similarity = np.exp(-distance)
    return similarity


def calculate_weights(matrix_list):
    """
    计算矩阵之间的权重，根据矩阵之间的相似度。
    """
    n_matrices = len(matrix_list)
    weights = np.zeros((n_matrices, n_matrices))

    for i in range(n_matrices):
        for j in range(i + 1, n_matrices):
            sim = similarity_measure(matrix_list[i], matrix_list[j])
            weights[i, j] = sim
            weights[j, i] = sim

    row_sums = weights.sum(axis=1)
    normalized_weights = row_sums / np.sum(row_sums)

    return normalized_weights


def fuse_adjacency_matrices_similarity_based(matrix_list):
    """
    融合邻接矩阵的相似度基准方法
    """
    n_matrices = len(matrix_list)
    weights = calculate_weights(matrix_list)

    fused_matrix = np.zeros(matrix_list[0].shape)

    for i in range(n_matrices):
        fused_matrix += weights[i] * matrix_list[i]

    # 使用中位数阈值化融合矩阵
    median_value = np.median(fused_matrix)
    binary_fused_matrix = (fused_matrix > median_value).astype(float)

    # 确保对角线元素为零
    np.fill_diagonal(binary_fused_matrix, 0)

    return binary_fused_matrix
