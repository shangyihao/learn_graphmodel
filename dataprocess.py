import os
import pickle
import urllib
import numpy as np
import itertools
import scipy.sparse as sp
from collections import namedtuple

Data = namedtuple('Data', ['x', 'y', 'adjacency',
                           'train_mask', 'val_mask', 'test_mask'])


class CoraData(object):
    download_url = "http://github.com/kimiyoung/planetoid/raw/master/data"
    filenames = ["ind.cora.{}".format(name) for name in
                 ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]

    def __init__(self, data_root="cora", rebuild=False):
        """
        包括数据下载、处理、加载等功能
        当数据的缓存文件存在时，使用缓存文件，否则将下载、处理，并缓存到磁盘

        :param data_root:string, optional
                存放数据的目录，原始数据路径：{data_root}/raw
                缓存数据路径：{data_root}/processed_cora.pkl
        :param rebuild: boolean, optional
                是否需要重新构建数据集，当设为True是，如果缓存数据存在也会重建数据
        """
        self.data_root = data_root
        save_file = os.path.join(self.data_root, "processed_cora.pkl")
        if os.path.exists(save_file) and not rebuild:
            print("Using Cached file: {}".format(save_file))
            self._data = pickle.load(open(save_file, "rb"))
        else:
            self.maybe_download()
            self._data = self.process_data()
            with open(save_file, "wb") as f:
                pickle.dump(self.data, f)
            print("Cached file: {}".format(save_file))

    @property
    def data(self):
        """返回Data数据对象，包括x，y, adjacency, train_mask, val_mask, test_mask"""
        return self._data

    def maybe_download(self):
        save_path = os.path.join(self.data_root, "raw")
        for name in self.filenames:
            if not os.path.exists(os.path.join(save_path, name)):
                self.download_data(
                    "{}/{}".format(self.download_url, name), save_path
                )

    @staticmethod
    def download_data(url, save_path):
        """数据下载工具，当原始数据不存在时将会进行下载"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data = urllib.request.urlopen(url)
        filename = os.path.basename(url)

        with open(os.path.join(save_path, filename), 'wb') as f:
            f.write(data.read())

        return True

    def process_data(self):
        """
        处理数据，得到节点特征和标签， 邻接矩阵，训练集、验证集以及测试机
        """
        print("Processing data...")
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(
            os.path.join(self.data_root, "raw", name)) for name in self.filenames]
        train_index = np.arange(y.shape[0])
        val_index = np.arange(y.shape[0], y.shape[0]+500)
        sorted_test_index = sorted(test_index)

        x = np.concatenate((allx, tx), axis=0)
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)

        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]
        num_nodes = x.shape[0]

        train_mask = np.zeros(num_nodes, dtype=np.bool)
        val_mask = np.zeros(num_nodes, dtype=np.bool)
        test_mask = np.zeros(num_nodes, dtype=np.bool)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True
        adjacency = self.build_adjacency(graph)
        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape: ", adjacency.shape)
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation node: ", val_mask.sum())
        print("Number of test node: ", test_mask.sum())

        return Data(x=x, y=y, adjacency=adjacency, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    @staticmethod
    def build_adjacency(adj_dict):
        """根据邻接表创建邻接矩阵"""
        edge_index = []
        num_nodes = len(adj_dict)
        for src, dst in adj_dict.items():
            edge_index.extend([src, v] for v in dst)
            edge_index.extend([v, src] for v in dst)
        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))  # 由于上述得到的结果中存在重复的边，删掉这些重复的边
        edge_index = np.asarray(edge_index)
        adjacency = sp.coo_matrix((np.ones(len(edge_index)),
                                   (edge_index[:, 0], edge_index[:, 1])),
                                  shape=(num_nodes, num_nodes), dtype="float32")
        return adjacency

    @staticmethod
    def read_data(path):
        """使用不同方式读取原始数据以进一步处理"""
        name = os.path.basename(path)
        if name == "ind.cora.test.index":
            out = np.genfromtxt(path, dtype="int64")
            return out
        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")
            out = out.toarray() if hasattr(out, "toarray") else out
            return out

