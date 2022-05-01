import torch
from tqdm import tqdm
from utils import SummaryWriter
from copy import deepcopy
from torch.autograd import Variable

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

class AutoPrepBatch(object):
    def __init__(self, prep_pipeline, model, loss_fn, model_optimizer, prep_pipeline_optimizer,
                model_scheduler, prep_pipeline_scheduler, params, log_dir=None):
        self.prep_pipeline = prep_pipeline
        self.model = model
        self.loss_fn = loss_fn
        self.model_optimizer = model_optimizer
        self.prep_pipeline_optimizer = prep_pipeline_optimizer
        self.model_scheduler = model_scheduler
        self.prep_pipeline_scheduler = prep_pipeline_scheduler
        self.params = params
        self.device = self.params["device"]
        self.writer = None

        if log_dir is not None:
            self.writer = SummaryWriter(log_dir)

    def forward_propagate(self, X, y, X_type, require_transform_grad=False,
                          require_model_grad=False, max_only=False):
        """ Forward pass"""
        with torch.set_grad_enabled(require_transform_grad):
            X_trans = self.prep_pipeline.transform(X, X_type=X_type, max_only=max_only, resample=False)

        if X_type == "train":
            self.model.train()
        else:
            self.model.eval()

        with torch.set_grad_enabled(require_model_grad or require_transform_grad):
            X_trans = X_trans.to(self.device)
            output = self.model(X_trans)
        y = y.to(self.device)
        loss = self.loss_fn(output, y)
        return output, loss

    def fit(self, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
        # best_val_acc = 0
        best_val_loss = float("inf")
        best_epoch = -1
        # fit prep_pipeline
        self.prep_pipeline.fit(X_train)

        # log prep_pipeline
        if self.writer is not None:
            self.log_prep_pipeline(global_step=-1)

        t = tqdm(range(self.params["num_epochs"]))
        # t = range(self.params["num_epochs"])
        best_prep_pipeline = None

        # start training
        for e in t:
            self.global_step = e
            # print("epoch:", e)
            tr_loss, tr_acc = self.train(X_train, y_train, X_val, y_val)
            # print(self.prep_pipeline.tf_prob_sample)

            if self.writer is not None:
                self.log_prep_pipeline(global_step=e)
            prep_pipeline = deepcopy(self.prep_pipeline.state_dict())
            val_loss, val_acc = self.evaluate(X_val, y_val, X_type='val', max_only=False)
            test_loss, test_acc = self.evaluate(X_test, y_test, X_type='test', max_only=False)

            # if val_acc > best_val_acc:
            if val_loss < best_val_loss:
                best_epoch = e
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_test_acc = test_acc
                best_tr_acc = tr_acc
                best_prep_pipeline = prep_pipeline

            model_lr = self.model_optimizer.param_groups[0]['lr']

            t.set_postfix(tr_loss=tr_loss, val_loss=val_loss, lr=model_lr)
            # print("tr loss:", tr_loss, "tr_acc", tr_acc, "val_loss", val_loss, "val_acc", val_acc)

            # scheduler 
            if self.model_scheduler is not None:
                self.model_scheduler.step(val_loss)

            if self.prep_pipeline_scheduler is not None:
                self.prep_pipeline_scheduler.step(val_loss)

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
            "best_val_loss": best_val_loss,
            "best_test_acc": best_test_acc,
            "best_epoch": best_epoch,
            "num_features": self.prep_pipeline.out_features
        }
        return result, best_prep_pipeline

    def train(self, X_train, y_train, X_val, y_val):
        # self.prep_pipeline.show_alpha()
        # one epoch training

        self.update_prep_pipeline(X_train, y_train, X_val, y_val)
        self.prep_pipeline.fit(X_train)
        # update model
        tr_loss, tr_correct = self.update_model(X_train, y_train)
        tr_acc = tr_correct / len(y_train)
        return tr_loss, tr_acc

    def evaluate(self, X, y, X_type, max_only=True):
        output, loss = self.forward_propagate(X, y, X_type=X_type, max_only=max_only)
        _, preds = torch.max(output, 1)
        correct = torch.sum(preds.cpu() == y)
        acc = correct.item() / len(y)
        return loss.item(), acc

    def update_model(self, X_train, y_train):
        self.model_optimizer.zero_grad()
        output_train, loss_train = self.forward_propagate(X_train, y_train, X_type='train',
                                                          require_model_grad=True)
        loss_train.backward()
        self.model_optimizer.step()
        _, preds = torch.max(output_train, 1)
        correct = torch.sum(preds.cpu() == y_train)
        return loss_train.item(), correct.item()

    def update_prep_pipeline(self, X_train, y_train, X_val, y_val):
        self.prep_pipeline_optimizer.zero_grad()

        dval_dalpha, dval_dw = self.compute_dval(X_train, y_train, X_val, y_val)
        hessian_product = self.compute_hessian_product(X_train, y_train, dval_dw)
        # print("dval/dalpha after: ", dval_dalpha)

        for i, (alpha, dval, dtrain) in enumerate(zip(self.prep_pipeline.parameters(), dval_dalpha, hessian_product)):
            dalpha = dval - self.model_optimizer.param_groups[0]['lr'] * dtrain
            # print(dval)
            # print(dtrain)
            if alpha.grad is None:
                alpha.grad = Variable(dalpha.data.clone())
            else:
                alpha.grad.data.copy_(dalpha.data.clone())

        # print("d_tf_prob_logits", self.prep_pipeline.tf_prob_logits[0].grad)
        self.prep_pipeline_optimizer.step()
        # print(self.prep_pipeline.tf_prob_logits[0])


    def compute_dval(self, X_train, y_train, X_val, y_val):
        model_backup = deepcopy(self.model.state_dict())
        # do virtual update on model using training data
        self.update_model(X_train, y_train)
        self.model_optimizer.zero_grad()
        output_val, loss_val = self.forward_propagate(X_val, y_val,
                                                      X_type='val',
                                                      require_transform_grad=True,
                                                      require_model_grad=True)

        # print("output val: ", output_val)
        # print("val loss: ", loss_val)
        loss_val.backward(retain_graph=True)
        # dval / dalpha
        dval_dalpha = [param.grad.data.clone() for param in self.prep_pipeline.parameters()]
        dval_dw = [param.grad.data.clone() for param in self.model.parameters()]
        # restore model
        self.model.load_state_dict(model_backup)
        # check whether loading state dict changes dval_dalpha and dval_dw
        return dval_dalpha, dval_dw

    def compute_hessian_product(self, X_train, y_train, dval_dw):
        model_backup = deepcopy(self.model.state_dict())
        # print(dval_dw)
        # raise
        eps = 0.001 * _concat(self.model.parameters()).data.detach().norm() / _concat(dval_dw).data.detach().norm()
        
        # print("eps: ", eps)
        # print(dval_dw)
        for w, dw in zip(self.model.parameters(), dval_dw):
            w.data += eps * dw
        # dtrain / dalpha
        output_train, loss_train = self.forward_propagate(X_train, y_train, X_type='train',
                                                          require_transform_grad=True)

        # print("loss", loss_train)

        grads_p = torch.autograd.grad(loss_train, self.prep_pipeline.parameters(), retain_graph=True, allow_unused=True)

        for w, dw in zip(self.model.parameters(), dval_dw):
            w.data -= 2 * eps * dw
        output_train, loss_train = self.forward_propagate(X_train, y_train, X_type='train',
                                                          require_transform_grad=True)
        grads_n = torch.autograd.grad(loss_train, self.prep_pipeline.parameters(), retain_graph=True, allow_unused=True)
        hessian_product = [(x - y).div_(2 * eps.cpu()) for x, y in zip(grads_p, grads_n)]
        self.model.load_state_dict(model_backup)

        # if self.global_step == 1:
        
        # print("Dval", dval_dalpha[1][4], hessian_product[1][4])
        # print("hessian", hessian_product[1][4])

        return hessian_product

    def log_prep_pipeline(self, global_step):
        self.writer.add_pipeline(self.prep_pipeline.pipeline, global_step)

    def train_network(self, prep_pipeline, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
        self.prep_pipeline.load_state_dict(prep_pipeline)
        X_train = self.prep_pipeline.get_final_dataset(X_train, "train")
        X_val = self.prep_pipeline.get_final_dataset(X_val, "val")
        X_test = self.prep_pipeline.get_final_dataset(X_test, "test")
        best_val_acc = 0
        t = tqdm(range(self.params["num_epochs"]))

        # start training
        for e in t:
            # print("epoch:", e)
            tr_loss, tr_acc = self.basic_train(X_train, y_train)
            val_loss, val_acc = self.basic_evaluate(X_val, y_val)
            test_loss, test_acc = self.basic_evaluate(X_test, y_test)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_tr_acc = tr_acc

            model_lr = self.model_optimizer.param_groups[0]['lr']
            t.set_postfix(tr_loss=tr_loss, val_loss=val_loss)
            # print("tr loss:", tr_loss, "tr_acc", tr_acc, "val_loss", val_loss, "val_acc", val_acc)

            if self.model_scheduler is not None:
                self.model_scheduler.step(val_loss)

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
            "best_test_acc": best_test_acc
        }
        return result

    def basic_train(self, X_train, y_train):
        self.model.train()
        output_train = self.model(X_train.to(self.device))
        loss_train = self.loss_fn(output_train, y_train.to(self.device))

        self.model_optimizer.zero_grad()
        loss_train.backward()
        self.model_optimizer.step()

        _, preds = torch.max(output_train, 1)
        correct = torch.sum(preds.cpu() == y_train)

        # update model
        tr_loss, tr_correct = loss_train.item(), correct.item()
        tr_acc = tr_correct / len(y_train)
        return tr_loss, tr_acc

    def basic_evaluate(self, X, y):
        self.model.eval()
        output = self.model(X.to(self.device))
        loss = self.loss_fn(output, y.to(self.device))
        _, preds = torch.max(output, 1)
        correct = torch.sum(preds.cpu() == y)
        acc = correct.item() / len(y)
        return loss.item(), acc