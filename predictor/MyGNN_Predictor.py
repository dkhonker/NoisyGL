from predictor.Base_Predictor import Predictor
from predictor.module.GNNs import GCN
import time
import torch
import torch.nn.functional as F
from copy import deepcopy
import torch_geometric.utils as utils


class mygnn_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):
        self.model = GCN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'],
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
        for epoch in range(self.conf.training['n_epochs']):

            improve = ''
            t0 = time.time()
            self.model.train()
            self.predictor.train()
            self.optim.zero_grad()

            # obtain representations and rec loss of the estimator
            features, adj = self.feats, self.adj
            edge_index = adj.indices()

            
            output = self.model(features, adj)
            pred_model = F.softmax(output, dim=1)

            eps = 1e-8
            pred_model = pred_model.clamp(eps, 1 - eps)

            

            # loss of GCN classifier
            loss_gcn = self.loss_fn(output[self.train_mask], self.noisy_label[self.train_mask])

            total_loss = loss_gcn

            total_loss.backward()

            self.optim.step()

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
                self.predictor_model_weigths = deepcopy(self.predictor.state_dict())
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
        self.predictor.eval()
        with torch.no_grad():
            features = self.feats
            with torch.no_grad():
                output = self.model(features, self.adj)
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
            output = self.model(features, adj)
            loss_test = self.loss_fn(output[idx_test], labels[idx_test])
            acc_test = self.metric(labels[idx_test].cpu().numpy(), output[idx_test].detach().cpu().numpy())
            if self.conf.training["debug"]:
                print("\tGCN classifier results:",
                      "loss= {:.4f}".format(loss_test.item()),
                      "accuracy= {:.4f}".format(acc_test.item()))

        return loss_test, acc_test
