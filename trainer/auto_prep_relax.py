import torch
import torch.nn as nn
from tqdm import tqdm
import math
import utils
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from copy import deepcopy
from torch.autograd import Variable
from utils import SummaryWriter
from torch.distributions.utils import logits_to_probs


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class AutoPrepRelax(object):
    def __init__(self, transformer, model, loss_fn, model_optimizer, transformer_optimizer,
                 model_scheduler, transformer_scheduler, params, log_dir=None):
        super(AutoPrepRelax, self).__init__()
        self.transformer = transformer
        self.model = model
        self.loss_fn = loss_fn
        self.model_optimizer = model_optimizer
        self.transformer_optimizer = transformer_optimizer
        self.model_scheduler = model_scheduler
        self.transformer_scheduler = transformer_scheduler
        self.params = params
        self.device = self.params["device"]
        self.writer = None
        self.bs = self.params["batch_size"]
        self.X_trans = None
        # self.end = -1

        if log_dir is not None:
            self.writer = SummaryWriter(log_dir)

    def forward_propogate(self, X, y, is_train=False, require_transform_grad=False,
                          require_model_grad=False, max_only=False, step=None):
        """ Forward pass"""
        with torch.set_grad_enabled(require_transform_grad):
            if step is not None:
                if step == 0:
                    self.X_trans = self.transformer.transform(X, max_only=max_only, resample=False)
                if (step + 1) * self.bs > X.shape[0]:
                    X_trans = self.X_trans[step * self.bs:, :]
                    y = y[step * self.bs:]
                    # self.end = 0
                else:
                    end = (step + 1) * self.bs
                    X_trans = self.X_trans[step * self.bs:end, :]
                    y = y[step * self.bs:end]
            else:
                X_trans = self.transformer.transform(X, max_only=max_only, resample=False)

        if is_train:
            self.model.train()
        else:
            self.model.eval()
        with torch.set_grad_enabled(require_model_grad or require_transform_grad):
            X_trans = X_trans.to(self.device)
            output = self.model(X_trans)
        # if step is not None:
        #     # print(self.end)
        #     # print(X_trans)
        #     print(X_trans.shape)
        #     # print(y)
        #     print(y.shape)
        y = y.to(self.device)
        loss = self.loss_fn(output, y)
        if step is not None:
            return output, loss, y.cpu()
        else:
            return output, loss

    def fit(self, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
        best_val_acc = 0
        best_epoch = -1
        # fit transformer
        self.transformer.fit(X_train)

        # log transformer
        if self.writer is not None:
            self.log_transformer(global_step=-1)
        t = tqdm(range(self.params["num_epochs"]))

        train_iter = math.ceil(X_train.shape[0] / self.params["batch_size"])
        test_iter = math.ceil(X_test.shape[0] / self.params["batch_size"])
        val_iter = math.ceil(X_val.shape[0] / self.params["batch_size"])

        # start training
        for e in t:

            # print("epoch:", e)
            tr_loss, tr_acc = self.train(X_train, y_train, X_val, y_val, train_iter, val_iter)
            # print(tr_loss, tr_acc)
            # print(self.transformer.tf_prob_sample)

            if self.writer is not None:
                self.log_transformer(global_step=e)

            val_loss, val_acc = self.evaluate(val_iter, X_val, y_val)
            test_loss, test_acc = self.evaluate(test_iter, X_test, y_test)

            if val_acc > best_val_acc:
                best_epoch = e
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_tr_acc = tr_acc

            model_lr = self.model_optimizer.param_groups[0]['lr']

            t.set_postfix(tr_loss=tr_loss, val_loss=val_loss, lr=model_lr)
            # print("tr loss:", tr_loss, "tr_acc", tr_acc, "val_loss", val_loss, "val_acc", val_acc)

            # scheduler
            if self.model_scheduler is not None:
                self.model_scheduler.step(val_loss)

            if self.transformer_scheduler is not None:
                self.transformer_scheduler.step(val_loss)

            # logging
            if self.writer is not None:
                self.writer.add_scalar('tr_loss', tr_loss, global_step=e)
                self.writer.add_scalar('tr_acc', tr_acc, global_step=e)
                self.writer.add_scalar('val_loss', val_loss, global_step=e)
                self.writer.add_scalar('val_acc', val_acc, global_step=e)
                self.writer.add_scalar('test_loss', test_loss, global_step=e)
                self.writer.add_scalar('test_acc', test_acc, global_step=e)
                self.writer.add_scalar('model_lr', model_lr, global_step=e)

        if self.writer is not None:
            self.writer.close()

        result = {
            "tr_loss": tr_loss,
            "tr_acc": tr_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_tr_acc": best_tr_acc,
            "best_val_acc": best_val_acc,
            "best_test_acc": best_test_acc,
            "best_epoch": best_epoch,
            "num_features": self.transformer.out_features
        }
        return result

    def train(self, X_train, y_train, X_val, y_val, train_iter, val_iter):
        self.update_transformer(X_train, y_train, X_val, y_val)  # this is BGD, is it better to find the next aplha?
        self.transformer.fit(X_train)
        n = 0
        tr_acc = 0
        tr_loss = 0
        # print("finish tf update")
        for step in range(train_iter):
            n += 1
            _tr_loss, tr_correct = self.update_model(X_train, y_train, step)  # 浼犲叆鐨勮繕鏄師濮嬬殑锛屼笉鏄痬ini-batch
            tr_loss += _tr_loss
            tr_acc += tr_correct / self.params["batch_size"]

        return tr_loss / n, tr_acc / n

    def evaluate(self, iter, X, y, max_only=True):
        loss = 0
        acc = 0
        n = 0
        for step in range(iter):
            n += 1
            output, _loss, _y = self.forward_propogate(X, y, max_only=max_only, step=step)
            _, preds = torch.max(output, 1)
            correct = torch.sum(preds.cpu() == _y)
            acc += correct.item() / len(_y)
            loss += _loss.item()
        return loss / n, acc / n

    def update_model(self, X_train, y_train, step=None):
        self.model_optimizer.zero_grad()
        if step is not None:
            output_train, loss_train, y_train = self.forward_propogate(X_train, y_train, is_train=True,
                                                                       require_model_grad=True, step=step)
            loss_train.backward(retain_graph=True)
        else:
            output_train, loss_train = self.forward_propogate(X_train, y_train, is_train=True,
                                                              require_model_grad=True, step=step)
            loss_train.backward()
        if self.params["grad_clip"] is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.params["grad_clip"])
        self.model_optimizer.step()
        _, preds = torch.max(output_train, 1)
        correct = torch.sum(preds.cpu() == y_train)
        return loss_train.item(), correct.item()

    def update_transformer(self, X_train, y_train, X_val, y_val):
        self.transformer_optimizer.zero_grad()

        dval_dalpha, dval_dw = self.compute_dval(X_train, y_train, X_val, y_val)
        hessian_product = self.compute_hessian_product(X_train, y_train, X_val, y_val, dval_dw)
        # print("dval/dalpha after: ", dval_dalpha)

        for i, (alpha, dval, dtrain) in enumerate(zip(self.transformer.augment_parameters(), dval_dalpha, hessian_product)):

            dalpha = dval - self.model_optimizer.param_groups[0]['lr'] * dtrain
            # print("dval: ", dval)
            # print("dtrain: ", dtrain)
            # print("dalpha: ", dalpha)
            if alpha.grad is None:
                alpha.grad = Variable(dalpha.data.clone())
            else:
                alpha.grad.data.copy_(dalpha.data.clone())
        # print("d_step_prob_logits", self.transformer.step_prob_logits.grad)
        # print("d_tf_prob_logits", self.transformer.tf_prob_logits[0].grad)
        self.transformer_optimizer.step()
        # print(self.transformer.augment_parameters())
        # print(self.transformer.tf_prob_logits[0])
        # print(self.transformer.step_prob_logits)

    def compute_dval(self, X_train, y_train, X_val, y_val):
        # for param in self.transformer.parameters():
        #     print(param.shape)
        model_backup = deepcopy(self.model.state_dict())
        # do virtual update on model using training data
        self.update_model(X_train, y_train)
        self.model_optimizer.zero_grad()
        output_val, loss_val = self.forward_propogate(X_val, y_val, require_transform_grad=True,
                                                      require_model_grad=True)
        # print("output val: ", output_val)
        # print("val loss: ", loss_val)
        loss_val.backward(retain_graph=True)
        dval_dalpha = self.transformer.relax(loss_val)
        dval_dw = [param.grad.data.clone() for param in self.model.parameters()]
        # restore model
        self.model.load_state_dict(model_backup)
        # check whether loading state dict changes dval_dalpha and dval_dw
        return dval_dalpha, dval_dw

    def compute_hessian_product(self, X_train, y_train, X_val, y_val, dval_dw):
        model_backup = deepcopy(self.model.state_dict())
        eps = 0.001 * _concat(self.model.parameters()).data.detach().norm() / _concat(dval_dw).data.detach().norm()
        # print("eps: ", eps)
        for w, dw in zip(self.model.parameters(), dval_dw):
            w.data += eps * dw
        # dtrain / dalpha
        output_train, loss_train = self.forward_propogate(X_train, y_train, is_train=True,
                                                          require_transform_grad=True)
        grads_p = self.transformer.relax(loss_train)
        for w, dw in zip(self.model.parameters(), dval_dw):
            w.data -= 2 * eps * dw
        output_train, loss_train = self.forward_propogate(X_train, y_train, is_train=True,
                                                          require_transform_grad=True)
        grads_n = self.transformer.relax(loss_train)
        hessian_product = [(x - y).div_(2 * eps.cpu()) for x, y in zip(grads_p, grads_n)]
        self.model.load_state_dict(model_backup)
        return hessian_product

    def log_transformer(self, global_step):
        for step in self.transformer.steps:
            step_name = step.name

            if not step.mandatory:
                step_probs = logits_to_probs(step.step_prob_logits.detach().data, is_binary=True).numpy()
                self.writer.add_step_probs('step {} step probs'.format(step_name), step_probs, global_step=global_step)

            if step.name != "first_step":
                tf_probs = logits_to_probs(step.tf_prob_logits.detach().data).numpy()
                self.writer.add_tf_probs('step {} tf probs'.format(step_name), tf_probs, global_step=global_step)
