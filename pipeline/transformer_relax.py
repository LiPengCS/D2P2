import torch
import torch.nn as nn
from copy import deepcopy
from torch.autograd import Variable
from torch.distributions.utils import logits_to_probs
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class QFunc(torch.nn.Module):
    '''Control variate for RELAX'''

    def __init__(self, num_latents, hidden_size=100):
        super(QFunc, self).__init__()
        # print(num_latents)
        self.h1 = torch.nn.Linear(num_latents, hidden_size)
        self.nonlin = torch.nn.Tanh()
        self.out = torch.nn.Linear(hidden_size, 1)

    def forward(self, p, w): # bernoulli - step - dim=1 ; categorical - tf - dim=2
        # the multiplication by 2 and subtraction is from toy.py...
        # it doesn't change the bias of the estimator, I guess
        # print(p, w)

        if p is not None and w is not None:
            if p.dim() > 1:
                p = p.reshape(-1)
                w = w.reshape(-1)
                # print("***")
                # print(p.shape)
                # print(w.shape)
                # print("***")
                z = torch.cat([p, w], dim=-1)
            else:
                z = torch.cat([p.unsqueeze(dim=-1), w], dim=-1)
            z = z.reshape(-1)
        elif p is None and w is not None:
            z = w
            z = z.reshape(-1)
        else:
            z = p
            if p.dim() > 1:
                z = z.reshape(-1)

        # z = self.h1(z * 2. - 1.)
        z = self.h1(z)
        z = self.nonlin(z)
        z = self.out(z)
        return z


class Step(nn.Module):
    """ One step in data preparation pipeline

    Params:
        tf_options (list): list of transformation functions.
    """

    def __init__(self, name, tf_options, mandatory=False, bernoulli=False):
        super(Step, self).__init__()
        self.name = name
        self.tf_options = tf_options
        self.num_options = len(tf_options)
        self.mandatory = mandatory
        self.bernoulli = bernoulli
        self.aug_parameters = []

    def init_parameters(self, in_features, type):
        self.type = type
        self.in_features = in_features
        # probs of whether to execute the step or not
        param_num = 0
        if not self.mandatory:
            self.step_prob_logits = nn.Parameter(0.5*torch.ones(self.in_features), requires_grad=True)
            self.aug_parameters.append(self.step_prob_logits)
            param_num += self.in_features
        else:
            self.step_prob_logits = None

        # probs of executing each tf in every step
        self.tf_prob_logits = nn.Parameter(torch.zeros(in_features, self.num_options), requires_grad=True)
        self.aug_parameters.append(self.tf_prob_logits)
        param_num += in_features * self.num_options
        # samples
        self.step_prob_sample = None
        self.tf_prob_sample = None  # shape (num_features, num_tfs)

        self.is_sampled = False
        if self.type == "relax":
            self.q_func = [QFunc(param_num)]  # seems input is the total number of latents
            self.aug_parameters += [*self.q_func[0].parameters()]

    def forward(self, X, is_train, max_only=False):
        # train tfs
        X_trans = []
        # print("step: ", step_prob_sample)
        # print("tf: ", tf_prob_sample)
        for tf in self.tf_options:
            if is_train:
                X_t = tf.fit_transform(X.detach().numpy())
            else:
                X_t = tf.transform(X.detach().numpy())

            X_t = torch.Tensor(X_t).unsqueeze(-1)
            X_trans.append(X_t)

        # All transformations
        X_trans = torch.cat(X_trans, dim=2)  # shape (num_examples, num_features, num_tfs)

        # select the sample from X transformations
        X_output = self.select_X_sample(X, X_trans, max_only)
        return X_output

    def select_X_sample(self, X, X_trans, max_only):
        if max_only:
            step_prob_sample, tf_prob_sample = self.sample_with_max_probs()
        else:
            step_prob_sample, tf_prob_sample = self.step_prob_sample, self.tf_prob_sample
        # print()
        X_trans_sample = (X_trans * tf_prob_sample.unsqueeze(0)).sum(axis=2)
        X_output = X_trans_sample * step_prob_sample.unsqueeze(0) + \
                   X * (1 - step_prob_sample.unsqueeze(0))
        return X_output

    def bernoulli_max(self, logits):
        max_sample = (logits > 0).int()
        return max_sample

    def categorical_max(self, logits):
        max_idx = torch.argmax(logits, dim=1)
        max_sample = torch.zeros_like(logits)
        max_sample[np.arange(max_sample.shape[0]), max_idx] = 1
        return max_sample

    def bernoulli_sample(self, logits, temperature, use_reparam=True):
        if not use_reparam:
            samples = logits # logits_to_probs(logits, is_binary=True)
        else:
            EPS = 1e-6
            logits = logits.clamp(0.0 + EPS, 1.0 - EPS)
            self.step_log_logits = torch.log(logits) - torch.log1p(-logits)
            u = torch.rand(logits.shape)
            u = u.clamp(EPS, 1.0)
            v = torch.rand(logits.shape)
            v = v.clamp(EPS, 1.0)
            z = self.step_log_logits + torch.log(u) - torch.log1p(-u)
            b = z.gt(0.0).type_as(z)
            samples = b

            def _get_probabilities_z_tilde(logits, b, v):
                theta = torch.sigmoid(logits)
                v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
                z_tilde = logits + torch.log(v_prime) - torch.log1p(-v_prime)
                return z_tilde

            z_tilde = _get_probabilities_z_tilde(self.step_log_logits, b, v)
            self.bernoulli_b = b
            self.bernoulli_z = torch.sigmoid(z / temperature)
            self.bernoulli_z_tilde = torch.sigmoid(z_tilde / temperature)
        return samples

    def categorical_sample(self, logits, temperature, use_reparam=True, first=False):
        if not use_reparam:
            samples = logits_to_probs(logits, is_binary=False)
        else:
            EPS = 1e-6
            self.tf_log_logits = torch.log(torch.nn.functional.softmax(logits, dim=-1))
            u = torch.rand(logits.shape)
            u = u.clamp(EPS, 1.0)
            v = torch.rand(logits.shape)
            v = v.clamp(EPS, 1.0)
            z = self.tf_log_logits - torch.log(-torch.log(u))
            b = torch.argmax(z, dim=-1)
            samples = torch.zeros(logits.shape, dtype=torch.int64)
            for i in range(b.shape[0]):
                samples[i, b[i]] = 1

            def _get_ops_weights_z_tilde(logits, b, v):
                theta = torch.exp(logits)
                vb = torch.log(v.gather(1, b.unsqueeze(-1)))
                z_tilde = -torch.log(-torch.log(v) / theta - vb)
                temp = -torch.log(-vb).squeeze()
                for i in range(z_tilde.shape[0]):
                    z_tilde[i][b[i]] = temp[i]

                return z_tilde

            z_tilde = _get_ops_weights_z_tilde(self.tf_log_logits, b, v)
            self.categorical_b = b
            self.categorical_z = torch.nn.functional.softmax(z / temperature, dim=-1)
            self.categorical_z_tilde = torch.nn.functional.softmax(z_tilde / temperature, dim=-1)
        if first:
            return samples, self.tf_log_logits, self.categorical_b, self.categorical_z, self.categorical_z_tilde
        else:
            return samples

    def sample(self, temperature, use_reparam=True):
        if self.mandatory:
            self.step_prob_sample = torch.ones(self.in_features)
            self.bernoulli_b, self.bernoulli_z, self.bernoulli_z_tilde = None, None, None
        else:
            self.step_prob_sample = self.bernoulli_sample(self.step_prob_logits, temperature, use_reparam) # [num_features]
        self.tf_prob_sample = self.categorical_sample(self.tf_prob_logits, temperature, use_reparam) # [num_features, tfs]
        self.is_sampled = True

    def sample_with_max_probs(self):
        if self.mandatory:
            step_prob_sample = torch.ones(self.in_features)
        else:
            step_prob_sample = self.bernoulli_max(self.step_prob_logits)
        tf_prob_sample = self.categorical_max(self.tf_prob_logits)
        return step_prob_sample, tf_prob_sample

    def show_alpha(self):
        print("step logits: ", self.step_prob_logits.data)
        print("tf logits", self.tf_prob_logits.data)

    def show_probs(self):
        print("step probs: ", logits_to_probs(self.step_prob_logits.data, is_binary=True))
        print("tf probs", logits_to_probs(self.tf_prob_logits.data))

    def relax(self, f_b):
        type = self.type
        if type == "relax":
            f_z = self.q_func[0](self.bernoulli_z, self.categorical_z) # [num_features] [num_features, tfs]
            f_z_tilde = self.q_func[0](self.bernoulli_z_tilde, self.categorical_z_tilde)

        if not self.mandatory:
            # if self.bernoulli:
            bernoulli_log_prob = torch.distributions.Bernoulli(logits=self.step_log_logits).log_prob(
                self.step_prob_sample) # [feature_num]
            # else:
            #     bernoulli_log_prob = torch.distributions.Bernoulli(logits=self.step_prob_logits).log_prob(
            #         self.step_prob_sample)  # [feature_num]
        else:
            bernoulli_log_prob = None
        temp = (self.tf_prob_sample == 1).nonzero()[:, 1].squeeze(0) # can't [num_features, tfs], need [num_features]
        categorical_log_prob = torch.distributions.Categorical(logits=self.tf_log_logits).log_prob(
            temp) # [feature_num, self.num_options] [feature_num]
        d_log_prob_list = []
        if type == "relax":
            diff = f_b.cpu() - f_z_tilde
        else:
            diff = f_b.cpu()
        if not self.mandatory:
            d_log_prob = torch.autograd.grad(
                [bernoulli_log_prob], [self.step_prob_logits], grad_outputs=torch.ones_like(bernoulli_log_prob),
                retain_graph=True)
            if type == "relax":
                d_f_z = torch.autograd.grad(
                    [f_z], [self.step_prob_logits], grad_outputs=torch.ones_like(f_z),
                    create_graph=True, retain_graph=True)
                d_f_z_tilde = torch.autograd.grad(
                    [f_z_tilde], [self.step_prob_logits], grad_outputs=torch.ones_like(f_z_tilde),
                    create_graph=True, retain_graph=True)
                d_log_prob_list.append(diff*d_log_prob[0].detach().data + d_f_z[0].detach().data - d_f_z_tilde[0].detach().data)
            else:
                d_log_prob_list.append(diff * d_log_prob[0].detach().data)
        d_log_prob = torch.autograd.grad(
            [categorical_log_prob], [self.tf_prob_logits], grad_outputs=torch.ones_like(categorical_log_prob),
            retain_graph=True)
        if type == "relax":
            d_f_z = torch.autograd.grad(
                [f_z], [self.tf_prob_logits], grad_outputs=torch.ones_like(f_z),
                create_graph=True, retain_graph=True)
            d_f_z_tilde = torch.autograd.grad(
                [f_z_tilde], [self.tf_prob_logits], grad_outputs=torch.ones_like(f_z_tilde),
                create_graph=True, retain_graph=True)
            d_log_prob_list.append(diff*d_log_prob[0].detach().data + d_f_z[0].detach().data - d_f_z_tilde[0].detach().data)
        else:
            d_log_prob_list.append(diff * d_log_prob[0].detach().data)
        if type == "relax":
            var_loss_list = ([d_logits ** 2 for d_logits in d_log_prob_list])
            if self.bernoulli_z is not None:
                var_loss = torch.cat([var_loss_list[0].unsqueeze(dim=-1), var_loss_list[1]], dim=-1).mean()
            else:
                var_loss = var_loss_list[0].mean()
            d_q_func = torch.autograd.grad(var_loss, self.q_func[0].parameters(), retain_graph=True)
            d_log_prob_list = d_log_prob_list + list(d_q_func)
        return d_log_prob_list


class FirstStep(Step):
    """" The first step in the pipeline. Cleaning missing values and one-hot encoding

    Params:
        tf_options: missing value imputers
    """

    def __init__(self, tf_options):
        super(FirstStep, self).__init__("first_step", tf_options, mandatory=True)
        self.tf_options = tf_options

    def fit(self, X, type):
        """ Train transformers and Initialize parameters

        Params:
            X (pd.DataFrame): numerical and categorical columns with missing values (np.nan)
        """
        self.type = type
        X_num = X.select_dtypes(include='number')
        X_cat = X.select_dtypes(exclude='number')

        X_num_trans = []
        X_cat_trans = []
        contain_num = X_num.shape[1] > 0
        contain_cat = X_cat.shape[1] > 0

        for tf in self.tf_options:
            if tf.input_type == "numerical" and contain_num:
                X_num_t = tf.fit_transform(X_num.values)
                X_num_trans.append(X_num_t)

            if tf.input_type == "categorical" and contain_cat:
                X_cat_t = tf.fit_transform(X_cat.values)
                X_cat_trans.append(X_cat_t)

            if tf.input_type == "mixed":
                X_num_t, X_cat_t = tf.fit_transform(X_num.values, X_cat.values)
                if contain_num:
                    X_num_trans.append(X_num_t)
                if contain_cat:
                    X_cat_trans.append(X_cat_t)

        # initialize parameters
        self.num_tf_prob_logits = None
        self.cat_tf_prob_logits = None
        self.in_features = 0
        param_num = 0
        if contain_num:
            self.num_tf_prob_logits = nn.Parameter(
                torch.zeros(X_num.shape[1], len(X_num_trans)), requires_grad=True)  # [feature_num, tf_num]
            self.in_features += X_num.shape[1]
            self.aug_parameters.append(self.num_tf_prob_logits)
            param_num += X_num.shape[1] * len(X_num_trans)

        if contain_cat:
            # fit one hot encoder on all results of X_cat
            self.one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
            X_cat_trans_concat = np.vstack(X_cat_trans)
            X_cat_trans_concat_enc = self.one_hot_encoder.fit_transform(X_cat_trans_concat)

            self.cat_tf_prob_logits = nn.Parameter(
                torch.zeros(X_cat_trans_concat_enc.shape[1], len(X_cat_trans)),
                requires_grad=True)  # [feature_num, tf_num]
            self.aug_parameters.append(self.cat_tf_prob_logits)
            self.in_features += X_cat_trans_concat_enc.shape[1]
            param_num += X_cat_trans_concat_enc.shape[1] * len(X_cat_trans)

        # print("init num_tf_prob: ", self.num_tf_prob_logits.shape)
        # print("init cat_tf_prob: ", self.cat_tf_prob_logits.shape)

        # samples
        self.num_tf_prob_sample = None  # shape [feature_num, tf_num]
        self.cat_tf_prob_sample = None

        # save numerical column indices (place num before cat)
        self.num_num_features = X_num.shape[1]
        if self.type == "relax":
            self.q_func = [QFunc(param_num)]
            self.aug_parameters += [*self.q_func[0].parameters()]
        # print("*********")
        # for p in self.q_func[0].parameters():
        #     print(p.shape)
        # print("********")

    def forward(self, X, is_train, max_only=False):
        """ Forward pass
        Params:
            X (pd.DataFrame): numerical and categorical columns with missing values
        """
        X_num = X.select_dtypes(include='number')
        X_cat = X.select_dtypes(exclude='number')

        X_num_trans = []
        X_cat_trans = []
        contain_num = X_num.shape[1] > 0
        contain_cat = X_cat.shape[1] > 0

        for tf in self.tf_options:
            if tf.input_type == "numerical" and contain_num:
                X_num_t = tf.transform(X_num.values)
                X_num_trans.append(X_num_t)

            if tf.input_type == "categorical" and contain_cat:
                X_cat_t = tf.transform(X_cat.values)
                X_cat_t = self.one_hot_encoder.transform(X_cat_t)
                X_cat_trans.append(X_cat_t)

            if tf.input_type == "mixed":
                X_num_t, X_cat_t = tf.transform(X_num.values, X_cat.values)
                if contain_num:
                    X_num_trans.append(X_num_t)
                if contain_cat:
                    X_cat_t = self.one_hot_encoder.transform(X_cat_t)
                    X_cat_trans.append(X_cat_t)

        # All transformations
        if contain_num:
            X_num_trans = torch.Tensor(np.array(X_num_trans)).permute(1, 2,
                                                                      0)  # shape (num_examples, num_features, num_tfs)

        if contain_cat:
            X_cat_trans = torch.Tensor(np.array(X_cat_trans)).permute(1, 2,
                                                                      0)  # shape (num_examples, num_features, num_tfs)

        # select the sample from X transformations
        X_output = self.select_X_sample(X_num_trans, X_cat_trans, max_only)
        return X_output

    def select_X_sample(self, X_num_trans, X_cat_trans, max_only):
        if max_only:
            num_tf_prob_sample, cat_tf_prob_sample = self.sample_with_max_probs()
        else:
            num_tf_prob_sample, cat_tf_prob_sample = self.num_tf_prob_sample, self.cat_tf_prob_sample

        if cat_tf_prob_sample is None:
            X_output = (X_num_trans * num_tf_prob_sample.unsqueeze(0)).sum(axis=2)
        elif num_tf_prob_sample is None:
            X_output = (X_cat_trans * cat_tf_prob_sample.unsqueeze(0)).sum(axis=2)
        else:
            X_num_trans_sample = (X_num_trans * num_tf_prob_sample.unsqueeze(0)).sum(axis=2)
            X_cat_trans_sample = (X_cat_trans * cat_tf_prob_sample.unsqueeze(0)).sum(axis=2)
            X_output = torch.cat((X_num_trans_sample, X_cat_trans_sample), dim=1)
        return X_output

    def sample_with_max_probs(self):
        num_tf_prob_sample = None
        cat_tf_prob_sample = None
        if self.num_tf_prob_logits is not None:
            num_tf_prob_sample = self.categorical_max(self.num_tf_prob_logits)
        if self.cat_tf_prob_logits is not None:
            cat_tf_prob_sample = self.categorical_max(self.cat_tf_prob_logits)
        return num_tf_prob_sample, cat_tf_prob_sample

    def sample(self, temperature=0.1, use_reparam=True):
        if self.num_tf_prob_logits is not None:
            self.num_tf_prob_sample, self.num_tf_log_logits, self.num_b, self.num_z, self.num_z_tilde = self.categorical_sample(self.num_tf_prob_logits, temperature, use_reparam, first=True)
        else:
            self.num_b, self.num_z, self.num_z_tilde = None, None, None
        if self.cat_tf_prob_logits is not None:
            self.cat_tf_prob_sample, self.cat_tf_log_logits, self.cat_b, self.cat_z, self.cat_z_tilde = self.categorical_sample(self.cat_tf_prob_logits, temperature, use_reparam, first=True)
        else:
            self.cat_b, self.cat_z, self.cat_z_tilde = None, None, None

        self.is_sampled = True

    def relax(self, f_b):
        type = self.type
        if type == "relax":
            f_z = self.q_func[0](self.num_z, self.cat_z)
            f_z_tilde = self.q_func[0](self.num_z_tilde, self.cat_z_tilde)
        num_prob = 0
        if self.num_tf_prob_logits is not None:
            temp = (self.num_tf_prob_sample == 1).nonzero()[:, 1].squeeze(0) # can't [num_features, tfs], need [num_features]
            categorical_log_prob = torch.distributions.Categorical(logits=self.num_tf_log_logits).log_prob(
                temp)
            num_prob = categorical_log_prob
        cat_prob = 0
        if self.cat_tf_prob_logits is not None:
            temp = (self.cat_tf_prob_sample == 1).nonzero()[:, 1].squeeze(0)
            categorical_log_prob = torch.distributions.Categorical(logits=self.cat_tf_log_logits).log_prob(
                temp)
            cat_prob = categorical_log_prob
        d_log_prob_list = []
        if type == "relax":
            diff = f_b.cpu() - f_z_tilde
        else:
            diff = f_b.cpu()

        if self.num_tf_prob_logits is not None:
            d_log_prob = torch.autograd.grad(
                [num_prob], [self.num_tf_prob_logits], grad_outputs=torch.ones_like(num_prob),
                retain_graph=True)
            if type == "relax":
                d_f_z = torch.autograd.grad(
                    [f_z], [self.num_tf_prob_logits], grad_outputs=torch.ones_like(f_z),
                    create_graph=True, retain_graph=True)
                d_f_z_tilde = torch.autograd.grad(
                    [f_z_tilde], [self.num_tf_prob_logits], grad_outputs=torch.ones_like(f_z_tilde),
                    create_graph=True, retain_graph=True)
                d_log_prob_list.append(
                    diff * d_log_prob[0].detach().data + d_f_z[0].detach().data - d_f_z_tilde[0].detach().data)
            else:
                d_log_prob_list.append(diff * d_log_prob[0].detach().data)
        if self.cat_tf_prob_logits is not None:
            d_log_prob = torch.autograd.grad(
                [cat_prob], [self.cat_tf_prob_logits], grad_outputs=torch.ones_like(cat_prob),
                retain_graph=True)
            if type == "relax":
                d_f_z = torch.autograd.grad(
                    [f_z], [self.cat_tf_prob_logits], grad_outputs=torch.ones_like(f_z),
                    create_graph=True, retain_graph=True)
                d_f_z_tilde = torch.autograd.grad(
                    [f_z_tilde], [self.cat_tf_prob_logits], grad_outputs=torch.ones_like(f_z_tilde),
                    create_graph=True, retain_graph=True)
                d_log_prob_list.append(
                    diff * d_log_prob[0].detach().data + d_f_z[0].detach().data - d_f_z_tilde[0].detach().data)
            else:
                d_log_prob_list.append(diff * d_log_prob[0].detach().data)
        if type == "relax":
            var_loss_list = ([d_logits ** 2 for d_logits in d_log_prob_list])
            if self.num_tf_prob_logits is not None and self.cat_tf_prob_logits is not None:
                # print(var_loss_list)
                var_loss = torch.cat([var_loss_list[0].reshape(-1), var_loss_list[1].reshape(-1)], dim=-1).mean()
            else:
                var_loss = var_loss_list[0].mean()
            d_q_func = torch.autograd.grad(var_loss, self.q_func[0].parameters(), retain_graph=True)
            d_log_prob_list = d_log_prob_list + list(d_q_func)
        return d_log_prob_list


class Pipeline(nn.Module):
    """ Data preparation pipeline"""

    def __init__(self, prep_steps_dict, temperature=0.1, use_cuda=False, use_reparam=True, type="relax"):
        super(Pipeline, self).__init__()
        steps = [FirstStep(prep_steps_dict[0]["tf_options"])]
        for step in prep_steps_dict[1:]:
            steps.append(Step(step["name"], step["tf_options"], mandatory=step["mandatory"]))
        self.steps = nn.ModuleList(steps)

        # initialize steps
        self.steps = nn.ModuleList(steps)
        if use_cuda:
            self.temperature = torch.tensor(temperature).cuda()
        else:
            self.temperature = torch.tensor(temperature)
        self.num_steps = len(steps)
        self.is_fitted = False
        self.use_reparam = use_reparam
        self.bernoulli = True
        self.aug_parameters = []
        self.type = type

    def augment_parameters(self):
        return self.aug_parameters

    def init_parameters(self, X_train):
        # initialize parameters in steps
        self.steps[0].fit(X_train, self.type)
        self.aug_parameters += self.steps[0].aug_parameters
        # print(self.steps[0].aug_parameters)
        # print("**********")
        num_features = self.steps[0].in_features
        for i in range(1, self.num_steps):
            self.steps[i].init_parameters(num_features, self.type)
            self.aug_parameters += self.steps[i].aug_parameters
        #     print(self.steps[i].aug_parameters)
        #     print("**********")
        self.out_features = num_features

    def forward(self, X, is_train, resample=False, max_only=False):
        X_output = deepcopy(X)

        for step in self.steps:
            # do sampling
            # print('doing ', step.name)
            if resample or not step.is_sampled:
                step.sample(temperature=self.temperature, use_reparam=self.use_reparam)

            # forward
            X_output = step(X_output, is_train, max_only=max_only)

            # print(X_output)
        return X_output

    def fit(self, X):
        self.is_fitted = True
        return self.forward(X, is_train=True, resample=True)

    def transform(self, X, max_only=False, resample=False):
        "max_only: only do step with prob > 0.5 and tf with maximum prob"

        if not self.is_fitted:
            raise Exception("transformer is not fitted")

        return self.forward(X, is_train=False, resample=resample, max_only=max_only)

    def relax(self, f_b):
        grad_tuple = ()

        for step in self.steps:
            grad_list = step.relax(f_b)
            grad_tuple += tuple(grad_list)
        # print(grad_tuple)
        # print(self.parameters())
        # for name, param in self.named_parameters():
        #     print(name)
        #     print(param)
        # raise
        return grad_tuple

# class Pipeline(nn.Module):
#     """ Data preparation pipeline"""
#
#     def __init__(self, steps, temperature=0.1, use_cuda=False, use_reparam=False, identity=False):
#         super(Pipeline, self).__init__()
#         self.steps = steps
#         self.use_reparam = use_reparam
#         if use_cuda:
#             self.temperature = torch.tensor(temperature).cuda()
#         else:
#             self.temperature = torch.tensor(temperature)
#
#         self.num_steps = len(steps)
#
#         # probs of whether to execute the step or not
#         if identity:
#             self.step_prob_logits = nn.Parameter(Variable(-1e9 * torch.ones(self.num_steps), requires_grad=True))
#         else:
#             step_prob_logits = torch.zeros(self.num_steps)
#             # self.step_prob_logits = nn.Parameter(Variable(torch.zeros(self.num_steps), requires_grad=True))
#             for i in range(self.num_steps):
#                 if self.steps[i].mandatory is True:
#                     step_prob_logits[i] += 1e9
#
#             # self.step_prob_logits[i] = tmp
#             self.step_prob_logits = nn.Parameter(Variable(step_prob_logits, requires_grad=True))
#
#         num_letents = self.num_steps
#         # probs of executing each tf in every step
#         tf_prob_logits_list = []
#         for step in self.steps:
#             tf_prob_logits = nn.Parameter(Variable(torch.zeros(step.num_options), requires_grad=True))
#             tf_prob_logits_list.append(tf_prob_logits)
#             num_letents += step.num_options
#         self.tf_prob_logits = nn.ParameterList(tf_prob_logits_list)
#
#         self.q_func = [QFunc(num_letents)] # seems input is the total number of latents
#
#         self.is_fitted = False
#         self.step_prob_sample = None # this seems equal to self.probabilities_b
#         self.step_prob_sample_logits = None
#         self.step_prob_sample_sig_z = None
#         self.step_prob_sample_sig_z_tilde = None
#
#         self.tf_prob_sample = None  # this seems equal to the list of self.ops_weights_b
#         self.tf_prob_sample_logits = None
#         self.tf_prob_sample_softmax_z = None
#         self.tf_prob_sample_softmax_z_tilde = None
#
#         # need to check these two are still one-hot or just a biggest index?
#         self._augment_parameters = [
#             self.step_prob_logits,
#             self.tf_prob_logits,
#         ]
#         self._augment_parameters += [*self.q_func[0].parameters()] # just add in case that self.q_func can't get gradient as param
#
#     def bernoulli_sample(self, logits):
#         EPS = 1e-6
#         probabilities_logits = torch.log(logits.clamp(0.0 + EPS, 1.0 - EPS)) - torch.log1p(
#             -logits.clamp(0.0 + EPS, 1.0 - EPS))
#         probabilities_u = torch.rand(self.num_steps)
#         probabilities_v = torch.rand(self.num_steps) # don't know if need to convert to gpu
#         probabilities_u = probabilities_u.clamp(EPS, 1.0)
#         probabilities_v = probabilities_v.clamp(EPS, 1.0)
#         probabilities_z = probabilities_logits + torch.log(probabilities_u) - torch.log1p(-probabilities_u)
#         probabilities_b = probabilities_z.gt(0.0).type_as(probabilities_z)
#
#         def _get_probabilities_z_tilde(logits, b, v):
#             theta = torch.sigmoid(logits)
#             v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
#             z_tilde = logits + torch.log(v_prime) - torch.log1p(-v_prime)
#             return z_tilde
#
#         probabilities_z_tilde = _get_probabilities_z_tilde(probabilities_logits, probabilities_b, probabilities_v)
#         self.step_prob_sample = probabilities_b  # this seems equal to self.probabilities_b
#         self.step_prob_sample_logits = probabilities_logits
#         self.step_prob_sample_sig_z = torch.sigmoid(probabilities_z / self.temperature)
#         self.step_prob_sample_sig_z_tilde = torch.sigmoid(probabilities_z_tilde / self.temperature)
#
#     def categorical_sample(self, logits):
#         if not self.use_reparam:
#             print("not use reparam")
#             return logits_to_probs(logits, is_binary=False)
#         EPS = 1e-6
#         ops_weights_p = torch.nn.functional.softmax(logits, dim=-1)
#         ops_weights_logits = torch.log(ops_weights_p)
#         ops_weights_u = torch.rand(logits.shape)
#         ops_weights_v = torch.rand(logits.shape)
#         ops_weights_u = ops_weights_u.clamp(EPS, 1.0)
#         ops_weights_v = ops_weights_v.clamp(EPS, 1.0)
#         ops_weights_z = ops_weights_logits - torch.log(-torch.log(ops_weights_u))
#         ops_weights_b = torch.argmax(ops_weights_z, dim=-1)
#
#         def _get_ops_weights_z_tilde(logits, b, v):
#             theta = torch.exp(logits)
#             z_tilde = -torch.log(-torch.log(v) / theta - torch.log(v[b]))
#             z_tilde = z_tilde.scatter(dim=-1, index=b, src=-torch.log(-torch.log(v[b])))
#             # v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
#             # z_tilde = logits + torch.log(v_prime) - torch.log1p(-v_prime)
#             return z_tilde
#
#         ops_weights_z_tilde = _get_ops_weights_z_tilde(ops_weights_logits, ops_weights_b, ops_weights_v)
#         ops_weights_softmax_z = torch.nn.functional.softmax(ops_weights_z / self.temperature, dim=-1)
#         ops_weights_softmax_z_tilde = torch.nn.functional.softmax(ops_weights_z_tilde / self.temperature, dim=-1)
#         return ops_weights_logits, ops_weights_b, ops_weights_softmax_z, ops_weights_softmax_z_tilde
#         # self.ops_weights_b = ops_weights_b
#         # self.ops_weights_softmax_z = torch.nn.functional.softmax(ops_weights_z / self.temperature, dim=-1)
#         # self.ops_weights_softmax_z_tilde = torch.nn.functional.softmax(ops_weights_z_tilde / self.temperature, dim=-1)
#
#     def forward(self, X, is_train, resample=False):
#         if resample or self.step_prob_sample is None:
#             self.bernoulli_sample(self.step_prob_logits)
#
#             # print("sample step results: ", self.step_prob_sample)
#
#         if resample or self.tf_prob_sample is None:
#             self.tf_prob_sample = []
#             self.tf_prob_sample_logits = []
#             self.tf_prob_sample_softmax_z = []
#             self.tf_prob_sample_softmax_z_tilde = []
#             for i in range(self.num_steps):
#                 logits, b, softmax_z, softmax_z_tilde = self.categorical_sample(self.tf_prob_logits[i])
#                 self.tf_prob_sample.append(b)
#                 self.tf_prob_sample_logits.append(logits)
#                 self.tf_prob_sample_softmax_z.append(softmax_z)
#                 self.tf_prob_sample_softmax_z_tilde.append(softmax_z_tilde)
#
#             # print("sample tfs results: ", self.tf_prob_sample)
#         print(self.tf_prob_sample)
#         X_output = deepcopy(X)
#
#         for i in range(self.num_steps):
#             step = self.steps[i]
#             step_prob = self.step_prob_sample[i]
#             tf_probs = self.tf_prob_sample[i]
#             X_trans = step(X_output, tf_probs, is_train)
#
#             if step.mandatory:
#                 X_output = X_trans
#             else:
#                 X_output = step_prob * X_trans + (1 - step_prob) * X_output
#
#         return X_output
#
#     def forward_max(self, X):
#         # only do step with prob > 0.5 and tf with maximum prob
#         step_sample = self.step_prob_logits
#         X_output = deepcopy(X)
#
#         for i in range(self.num_steps):
#             step = self.steps[i]
#             step_prob = step_sample[i]
#             if step_prob < 0:
#                 continue
#             # print("do step ", i)
#             max_idx = torch.argmax(self.tf_prob_logits[i], dim=0)
#             # print("select max tf idx: ", max_idx)
#             tf_probs = torch.zeros_like(self.tf_prob_logits[i])
#             tf_probs[max_idx] = 1
#             X_trans = step(X_output, tf_probs, is_train=False)
#             X_output = X_trans
#
#         return X_output
#
#     def fit(self, X):
#         self.is_fitted = True
#         return self.forward(X, is_train=True, resample=True)
#
#     def transform(self, X, max_only=False, resample=False):
#         "max_only: only do step with prob > 0.5 and tf with maximum prob"
#
#         if not self.is_fitted:
#             raise Exception("transformer is not fitted")
#
#         if max_only:
#             return self.forward_max(X)
#         else:
#             return self.forward(X, is_train=False, resample=resample)
#
#     def show_alpha(self):
#         print("step logits: ", self.step_prob_logits.data)
#         for tf_prob in self.tf_prob_logits:
#             print("tf logits", tf_prob.data)
#
#     def show_probs(self):
#         print("step probs: ", logits_to_probs(self.step_prob_logits.data, is_binary=True))
#         for tf_prob in self.tf_prob_logits:
#             print("tf probs", logits_to_probs(tf_prob.data))
#
#     def relax(self, f_b):
#         f_z = self.q_func[0](self.step_prob_sample_sig_z, self.tf_prob_sample_softmax_z)
#         f_z_tilde = self.q_func[0](self.probabilities_sig_z_tilde, self.tf_prob_sample_softmax_z_tilde)
#         probabilities_log_prob = torch.distributions.Bernoulli(logits=self.step_prob_sample_logits).log_prob(
#             self.step_prob_sample)
#         log_prob = probabilities_log_prob
#         for i in range(self.num_steps):
#             ops_weights_log_prob = torch.distributions.Categorical(logits=self.tf_prob_sample_logits[i]).log_prob(
#                 self.tf_prob_sample[i])
#             log_prob += ops_weights_log_prob
#
#         d_log_prob_list = torch.autograd.grad(
#             [log_prob], [self.step_prob_logits, self.tf_prob_logits], grad_outputs=torch.ones_like(log_prob),
#             retain_graph=True)
#         d_f_z_list = torch.autograd.grad(
#             [f_z], [self.step_prob_logits, self.tf_prob_logits], grad_outputs=torch.ones_like(f_z),
#             create_graph=True, retain_graph=True)
#         d_f_z_tilde_list = torch.autograd.grad(
#             [f_z_tilde], [self.step_prob_logits, self.tf_prob_logits], grad_outputs=torch.ones_like(f_z_tilde),
#             create_graph=True, retain_graph=True)
#         diff = f_b - f_z_tilde
#         d_logits_list = [diff * d_log_prob + d_f_z - d_f_z_tilde for
#                          (d_log_prob, d_f_z, d_f_z_tilde) in zip(d_log_prob_list, d_f_z_list, d_f_z_tilde_list)]
#         # print([d_logits.shape for d_logits in d_logits_list])
#         var_loss_list = ([d_logits ** 2 for d_logits in d_logits_list])
#         # print([var_loss.shape for var_loss in var_loss_list])
#         var_loss = torch.cat([var_loss_list[0], var_loss_list[1].unsqueeze(dim=-1)], dim=-1).mean()
#         # var_loss.backward()
#         d_q_func = torch.autograd.grad(var_loss, self.q_func[0].parameters(), retain_graph=True)
#         d_logits_list = d_logits_list + list(d_q_func)
#         return [d_logits.detach() for d_logits in d_logits_list]