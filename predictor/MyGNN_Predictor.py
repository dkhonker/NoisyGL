from predictor.Base_Predictor import Predictor
from predictor.module.GNNs import GCN,GCNPlus,GCNPlus1
from torch.nn import Parameter
import math
import time
import torch
import torch.nn.functional as F
from copy import deepcopy
import torch.nn as nn
import torch_geometric.utils as utils
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add
import numpy as np

class mygnn_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):
        self.model = GCNPlus(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'],
                         out_channels=conf.model['n_classes'],
                         n_layers=conf.model['n_layer'], dropout=conf.model['dropout'],
                         norm_info=conf.model['norm_info'],
                         act=conf.model['act'], input_layer=conf.model['input_layer'],
                         output_layer=conf.model['output_layer']).to(self.device)


        self.optim = torch.optim.Adam(
            list(self.model.parameters()),
            lr=self.conf.training['lr'],
            weight_decay=self.conf.training['weight_decay'])

        # NRGNN
        self.best_pred = None
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_acc_pred_val = 0
        edge_index = self.adj.indices()
        features = self.feats
        self.edge_index = edge_index.to(self.device)
        self.idx_unlabel = torch.LongTensor(list(set(range(features.shape[0])) - set(self.train_mask))).to(self.device)


    def train(self):
      # features, adj = self.feats, self.adj
      # edge_index = adj.indices()
      # encoder = Encoder(in_channels=features.shape[1],out_channels=512, hidden_channels=1024, activation=F.relu,base_model=G2RGCNConv, k=1).to(self.device)
      # model = Model(encoder=encoder).to(self.device)        
      # coding_rate_loss = MaximalCodingRateReduction(gam1=0.5, gam2=0.5, eps=0.05).to(self.device)
      # optimizer = torch.optim.Adam( list(model.parameters()) + list(coding_rate_loss.parameters()), lr=0.001, weight_decay=0.0001)
      # for epoch in range(1, 21):
      #     model.train()
      #     optimizer.zero_grad()
      #     adj_dropped = random_edge_dropout(adj, drop_rate=0.3)
      #     features_shuffled = random_feature_shuffle(features, self.train_mask,shuffle_prob=0.0)
      #     z = model(features_shuffled, adj_dropped.indices())
      #     loss = coding_rate_loss(z, adj_dropped.to_dense())
      #     loss.backward()
      #     optimizer.step()
      # model.eval()
      # z = model(features, adj.indices())
      z = torch.randn(512,512).cuda()
      fc=nn.Linear(512,7,bias=False).to(self.device)
      optimizer = torch.optim.Adam(list(fc.parameters()),lr=0.01, weight_decay=0.0001)

      for epoch in range(self.conf.training['n_epochs']):
          improve = ''
          t0 = time.time()
          self.model.train()
          self.optim.zero_grad()

          fc.train()
          optimizer.zero_grad()
          
          # obtain representations and rec loss of the estimator
          features, adj = self.feats, self.adj
          edge_index = adj.indices()

          
          adj_dropped = random_edge_dropout(adj, drop_rate=0.3)
          features_shuffled = random_feature_shuffle(features, self.train_mask,shuffle_prob=0.0)
          output, output1 = self.model(features, adj_dropped)
          
          pred_model = F.softmax(output, dim=1)

          eps = 1e-8
          pred_model = pred_model.clamp(eps, 1 - eps)
          # loss of GCN classifier
          # if epoch >10:
          #   tmp = self.loss_fn(output[self.train_mask], self.noisy_label[self.train_mask], reduction='none')
          #   select_mask=self.train_mask[tmp.detach().cpu().numpy()<0.5]
          # else:
          select_mask=self.train_mask
          loss_gcn = self.loss_fn(output[select_mask], self.noisy_label[select_mask])
          loss_pse = self.loss_fn(output1[select_mask], self.noisy_label[select_mask])


          tmp = self.loss_fn(output[self.val_mask], self.noisy_label[self.val_mask], reduction='none')

          idx_add = self.val_mask[tmp.detach().cpu().numpy()<0.4]

          loss_add = self.loss_fn(output1[idx_add],F.one_hot(output[idx_add].max(dim=1)[1], 7).float())
          loss_add = nn.MSELoss()(F.one_hot(output[idx_add].max(dim=1)[1], 7).float(),output1[idx_add])
          
          
          loss_g2r = self.loss_fn(fc(z[select_mask].detach()),output[select_mask])

          total_loss = loss_gcn+\
                  1.5*loss_pse+\
                  0.0*loss_add+\
                  0.0*loss_g2r
          # total_loss = loss_gcn+\
          #         1.2*loss_pse+\
          #         0.1*loss_add+\
          #         0.0*loss_g2r

          total_loss.backward()

          self.optim.step()
          optimizer.step()

          # forward and backward
          acc_train = self.metric(self.noisy_label[self.train_mask].cpu().numpy(),
                                  output[self.train_mask].detach().cpu().numpy())

          # Evaluate validation set performance separately
          # acc_pred_val, acc_val, loss_val = self.evaluate(self.noisy_label[self.val_mask], self.val_mask, pred_edge_index, predictor_weights_1, model_edge_index, estimated_weights_1)
          loss_val, acc_val = self.evaluate(self.noisy_label[self.val_mask], self.val_mask)

          flag, flag_earlystop = self.recoder.add(loss_val, acc_val)
          if flag:
              improve = '*'
              self.total_time = time.time() - self.start_time
              self.best_val_loss = loss_val
              self.result['valid'] = acc_val
              self.result['train'] = acc_train

              self.best_val_acc = acc_val
              self.weights = deepcopy(self.model.state_dict())

              # self.best_acc_pred_val = acc_pred_val
          elif flag_earlystop:
              break

          if self.conf.training['debug']:
              print(
                  "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                      epoch + 1, time.time() - t0, total_loss.item(), acc_train, loss_val, acc_val, improve))

      print('Optimization Finished!')
      print('Time(s): {:.4f}'.format(self.total_time))
      loss_test, acc_test = self.test(self.test_mask)
      self.result['test'] = acc_test
      print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
      return self.result

    def evaluate(self, label, mask):

        self.model.eval()
        with torch.no_grad():
            features = self.feats
            with torch.no_grad():
                output,_ = self.model(features, self.adj)
            logits = output[mask]
            loss_val = self.loss_fn(logits, label)

            acc_val = self.metric(label.cpu().numpy(), output[mask].detach().cpu().numpy())
        return loss_val, acc_val

    def test(self, mask):
        features, adj = self.feats, self.adj
        labels = self.clean_label
        edge_index = self.edge_index
        idx_test = mask

        with torch.no_grad():
            self.model.eval()
            self.model.load_state_dict(self.weights)
            output,_ = self.model(features, adj)
            loss_test = self.loss_fn(output[idx_test], labels[idx_test])
            acc_test = self.metric(labels[idx_test].cpu().numpy(), output[idx_test].detach().cpu().numpy())
            if self.conf.training["debug"]:
                print("\tGCN classifier results:",
                      "loss= {:.4f}".format(loss_test.item()),
                      "accuracy= {:.4f}".format(acc_test.item()))

        return loss_test, acc_test

def random_edge_dropout(adj, drop_rate=0.2):
    # 获取稀疏张量的索引和数据
    indices = adj.coalesce().indices()
    values = adj.coalesce().values()

    # 计算每条边独立丢失的掩码
    drop_mask = torch.rand(values.size()) > drop_rate

    # 应用掩码
    new_indices = indices[:, drop_mask]
    new_values = values[drop_mask]

    new_adj = torch.sparse_coo_tensor(new_indices, new_values, adj.size())
    return new_adj.coalesce()  # 确保邻接矩阵是规范形式

def random_feature_shuffle(features, idx_train, shuffle_prob=0.2):
    # 对训练集中所有节点进行遍历
    for i in idx_train:
        if torch.rand(1).item() < shuffle_prob:  # 以 shuffle_prob 概率进行打乱
            # 随机选择另一个训练集节点
            swap_idx = idx_train[torch.randint(0, idx_train.shape[0], (1,))]
            # 交换特征
            features[i], features[swap_idx] = features[swap_idx].clone(), features[i].clone()
    return features

class MaximalCodingRateReduction(torch.nn.Module):
    ## This function is based on https://github.com/ryanchankh/mcr2/blob/master/loss.py
    def __init__(self, gam1=1.0, gam2=1.0, eps=0.01):
        super(MaximalCodingRateReduction, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=0)
        return z

    def compute_discrimn_loss_empirical(self, W):
        """Empirical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_empirical_all(self, W, Pi):
        """Empirical Compressive Loss."""
        p, m = W.shape
        k, _ = Pi.shape
        sum_trPi = torch.sum(Pi)

        I = torch.eye(p).cuda()
        compress_loss = 0.
        for j in range(k):
            trPi = torch.sum(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            a = W.T * Pi[j].view(-1, 1)
            a = a.T
            log_det = torch.logdet(I + scalar * a.matmul(W.T))
            compress_loss += log_det * trPi / m
        num = p
        compress_loss = compress_loss / 2 * (num / sum_trPi)
        return compress_loss

    def forward(self, X, A):
        i = np.random.randint(A.shape[0], size=768)
        A = A[i,::]
        A = A.cpu().numpy()
        W = X.T
        Pi = A
        Pi = torch.tensor(Pi, dtype=torch.float32).cuda()

        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
        compress_loss_empi = self.compute_compress_loss_empirical_all(W, Pi)
        total_loss_empi = - self.gam2 * discrimn_loss_empi + compress_loss_empi
        return total_loss_empi
    


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
    




class G2RGCNConv(MessagePassing):

    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 normalize=True,
                 **kwargs):
        super(G2RGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(self.node_dim), edge_weight, self.improved, x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
class Encoder(torch.nn.Module):
  def __init__(self,
                in_channels: int,
                hidden_channels: int,
                out_channels: int,
                activation,
                base_model=G2RGCNConv,
                k: int = 2):
      super(Encoder, self).__init__()
      self.base_model = base_model
      self.k = k
      self.conv_1 = self.base_model(in_channels,  hidden_channels)
      self.conv_0 = self.base_model(in_channels,  out_channels)

      for i in range(2, 10):
          exec("self.conv_%s = self.base_model( hidden_channels, hidden_channels )" % i)

      self.conv_last_layer = self.base_model(hidden_channels, out_channels)
      self.conv_layers_list = [self.conv_1, self.conv_2, self.conv_3]
      self.conv_layers_list.append(self.conv_last_layer)
      self.activation = activation
      self.prelu = nn.PReLU(out_channels)
      self.lin0 = nn.Linear(in_channels, hidden_channels, bias=True)
      self.lin1 = nn.Linear(hidden_channels, hidden_channels, bias=True)
      self.lin2 = nn.Linear(hidden_channels, hidden_channels, bias=True)
      self.lin3 = nn.Linear(hidden_channels, out_channels, bias=True)

  def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
      if self.k == 0:
          x = self.conv_0( x, edge_index )
          x = F.normalize(x, p=1)
          return x
      for i in range(0, self.k):
          x = self.activation(self.conv_layers_list[i](x, edge_index))
      x = self.conv_last_layer(x, edge_index)
      x = F.normalize(x, p=1)
      return x


class Model(torch.nn.Module):
  def __init__(self,encoder: Encoder):
      super(Model, self).__init__()
      self.encoder: Encoder = encoder

  def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
      return self.encoder(x, edge_index)