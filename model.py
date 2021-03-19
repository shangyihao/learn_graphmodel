import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import scipy.sparse as sp
import numpy as np
from dataprocess import CoraData


class GraphConvelutional(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """
        图卷积： L*X*\theta

        :param input_dim: int 节点输入特征的维度
        :param output_dim: int 输出特征的维度
        :param use_bias: bool,optional  是否使用偏置
        """
        super(GraphConvelutional, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """
        邻接矩阵是稀疏矩阵，因此在计算式使用稀疏矩阵乘法

        :param adjacency: torch.sparse.FloatTensor  邻接矩阵
        :param input_feature: torch.Tensor   输入特征
        """
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output


class GcnNet(nn.Module):
    """
    定义一个包含两层GraphConvelution的模型
    """
    def __init__(self, input_dim=1433):
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConvelutional(input_dim, 16)
        self.gcn2 = GraphConvelutional(16, 7)

    def forward(self, adjacency, feature):
        h = F.relu(self.gcn1(adjacency, feature))
        logits = self.gcn2(adjacency, h)
        return logits


def normalization(adjacency):
    """ 计算 L=D^-0.5 * (A+I) * D^-0.5"""
    adjacency += sp.eye(adjacency.shape[0])  # 增加自连接
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    return d_hat.dot(adjacency).dot(d_hat).tocoo()


# 超参数定义
learning_rate = 0.1
weight_decay = 5e-4
epochs = 200
# 模型定义，包括模型实例化、损失函数和优化器定义
device = 'cuda' if torch.cuda.is_available() else "cpu"
model = GcnNet().to(device)
# 损失函数使用交叉熵
criterion = nn.CrossEntropyLoss().to(device)
# 优化器使用Adam
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# 加载数据，并转换成torch.Tensor
dataset = CoraData().data
x = dataset.x / dataset.x.sum(1, keepdims=True)  # 归一化数据使得每一行和为1
tensor_x = torch.from_numpy(x).to(device)
tensor_y = torch.from_numpy(dataset.y).to(device)
tensor_train_mask = torch.from_numpy(dataset.train_mask).to(device)
tensor_val_mask = torch.from_numpy(dataset.val_mask).to(device)
tensor_test_mask = torch.from_numpy(dataset.test_mask).to(device)
normalize_adjacency = normalization(dataset.adjacency)  # 规范化邻接矩阵
indices = torch.from_numpy(
    np.array([normalize_adjacency.row,
              normalize_adjacency.col]).astype('int64')).long()
values = torch.from_numpy(normalize_adjacency.data.astype(np.float32))
tensor_adjacency = torch.sparse.FloatTensor(indices, values, (2708,2708)).to(device)


def train():
    loss_history = []
    val_acc_history = []
    model.train()
    train_y = tensor_y[tensor_train_mask]
    for epoch in range(epochs):
        logits = model(tensor_adjacency, tensor_x)  # 向前传播
        train_mask_logits = logits[tensor_train_mask]  # 只选择训练节点进行监督
        loss =criterion(train_mask_logits, train_y)  # 计算损失值
        optimizer.zero_grad()
        loss.backward()  # 反向传播计算参数的梯度
        optimizer.step()  # 使用优化方法进行梯度更新
        train_acc = test(tensor_train_mask)  # 计算当前模型在训练集上的准确率
        val_acc = test(tensor_val_mask)  # 计算当前模型在验证集上的准确率
        # 记录训练过程中损失值和准确率的变化
        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4f}, Valacc {:.4f}".format(
            epoch, loss.item(), train_acc.item(), val_acc.item()
        ))
    return loss_history, val_acc_history


def test(mask):
    model.eval()
    with torch.no_grad():
        logits = model(tensor_adjacency, tensor_x)
        test_mask_logits = logits[mask]
        predict_y = test_mask_logits.max(1)[1]
        accuracy = torch.eq(predict_y, tensor_y[mask]).float().mean()
    return accuracy


train()
test_acc = test(tensor_test_mask)
print("Test accuarcy: {}".format(test_acc))
