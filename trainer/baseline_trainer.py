import torch
from tqdm import tqdm
from utils import SummaryWriter

class BaselineTrainer(object):
    def __init__(self, model, loss_fn, model_optimizer, model_scheduler, params, log_dir=None):
        self.model = model
        self.loss_fn = loss_fn
        self.model_optimizer = model_optimizer
        self.params = params
        self.device = self.params["device"]
        self.writer = None
        self.model_scheduler = model_scheduler

        if log_dir is not None:
            self.writer = SummaryWriter(log_dir)

    def fit(self, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
        # best_val_acc = 0
        best_val_loss = float("inf")
        # t = tqdm(range(self.params["num_epochs"]))
        t = range(self.params["num_epochs"])

        # start training
        for e in t:
            # print("epoch:", e)
            tr_loss, tr_acc = self.train(X_train, y_train)
            val_loss, val_acc = self.evaluate(X_val, y_val)
            test_loss, test_acc = self.evaluate(X_test, y_test)

            # if val_acc > best_val_acc:
            if val_loss < best_val_loss:
                best_epoch = e
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_test_acc = test_acc
                best_tr_acc = tr_acc

            model_lr = self.model_optimizer.param_groups[0]['lr']
            # t.set_postfix(tr_loss=tr_loss, val_loss=val_loss)
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
            "best_epoch": best_epoch,
            "tr_loss": tr_loss,
            "tr_acc": tr_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_tr_acc": best_tr_acc,
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "best_test_acc": best_test_acc
        }
        return result

    def train(self, X_train, y_train):
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

    def evaluate(self, X, y):
        self.model.eval()
        output = self.model(X.to(self.device))
        loss = self.loss_fn(output, y.to(self.device))
        _, preds = torch.max(output, 1)
        correct = torch.sum(preds.cpu() == y)
        acc = correct.item() / len(y)
        return loss.item(), acc

