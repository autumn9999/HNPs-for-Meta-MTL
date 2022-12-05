import pdb

import torch
import torch.nn as nn

from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import basic_model

class MTNP(nn.Module):
    def __init__(self, config):
        super(MTNP, self).__init__()

        self.dataset = config["dset_name"]
        self.num_task = config["num_task"]
        self.way_number = config["way_number"]
        self.shot_number = config["shot_number"]
        self.w_repeat = config["w_repeat"]
        self.a_repeat = config["a_repeat"]

        # hyper-parameters related to our architecture
        if self.dataset == "domainnet":
            self.x_feature = 512
        else:
            self.x_feature = 4096
        model = basic_model

        # feature augmentation
        self.feature_extractor_index = False
        if self.feature_extractor_index:
            self.d_feature = 256
            self.feature_extractor = model.feature_extractor(self.x_feature, self.d_feature)
        else:
            self.d_feature = self.x_feature

        config["d_feature"] = self.d_feature

        # task-wise and class-wise transformers
        self.task_probabilistic_encoder = model.ProbabilisticEncoder_alpha(config, self.d_feature, self.num_task)
        self.class_probabilistic_encoder = model.ProbabilisticEncoder_phi(config, self.d_feature, self.way_number)

    @staticmethod
    def print_parameters(net):
        for name, parameters in net.named_parameters():
            print(name, ':', parameters.size())

    def task_wise_transformer_inference(self, x_c_order, x_t_order):

        task_prior = x_c_order.view(self.num_task, -1, self.d_feature)
        task_posterior = x_t_order.view(self.num_task, -1, self.d_feature)

        # infer for prior
        alpha_pmu, alpha_psigma = self.task_probabilistic_encoder(task_prior)
        alpha_pdistirbution = Normal(alpha_pmu, alpha_psigma)
        a_psample = alpha_pdistirbution.rsample([self.a_repeat])  # a_repeat * T * D

        # infer for posterior
        alpha_qmu, alpha_qsigma = self.task_probabilistic_encoder(task_posterior)
        alpha_qdistirbution = Normal(alpha_qmu, alpha_qsigma)
        a_qsample = alpha_qdistirbution.rsample([self.a_repeat])  # a_repeat * T * D

        # kl_alpha
        kl_a = kl_divergence(alpha_qdistirbution, alpha_pdistirbution).sum(dim=1)
        return kl_a, a_psample, a_qsample

    def class_wise_transformer_inference(self, a_qsample, a_psample, x_c_order, x_t_order, x_t):

        kl_w = []
        output_prior_list = []
        output_posterior_list = []
        for num in range(self.num_task):

            # input from the first layer--------------------------------------
            if self.training:
                task_embedding = a_qsample[:, num, :]
            else:
                task_embedding = a_psample[:, num, :]


            # input from the context-----------------------------------------
            class_prior = x_c_order
            class_posterior = x_t_order

            # infer for prior
            phi_pmu, phi_psigma = self.class_probabilistic_encoder(class_prior, task_embedding)
            phi_pdistirbution = Normal(phi_pmu, phi_psigma)

            # infer for posterior
            phi_qmu, phi_qsigma = self.class_probabilistic_encoder(class_posterior, task_embedding)
            phi_qdistirbution = Normal(phi_qmu, phi_qsigma)


            # task-specific kl_w
            task_specific_kl_w = kl_divergence(phi_qdistirbution, phi_pdistirbution).mean(dim=0).sum()
            kl_w.append(task_specific_kl_w.view(1))

            # =================make predictions========================================
            # ( way_number * num_target) * D
            predict_samples = x_t[num].contiguous().view(-1, self.d_feature)

            # w_repeat*a_repeat * ( way_number * num_target) * D
            repeat_predict_samples = predict_samples.unsqueeze(0).expand(self.w_repeat * self.a_repeat, predict_samples.shape[0],predict_samples.shape[1]).contiguous()

            if self.training:
                phi_qsample = phi_qdistirbution.rsample([self.w_repeat])
                phi_qsample = phi_qsample.transpose(0, 1).reshape(self.way_number, -1, self.d_feature)
                classifier_q = phi_qsample.transpose(0, 1).transpose(1, 2)

                phi_psample = phi_pdistirbution.rsample([self.w_repeat])
                phi_psample = phi_psample.transpose(0, 1).reshape(self.way_number, -1, self.d_feature)
                classifier_p = phi_psample.transpose(0, 1).transpose(1, 2)
            else:
                # -----using mean
                classifier_q = phi_qmu.unsqueeze(1).repeat(1, self.w_repeat, 1, 1).reshape(self.way_number, -1, self.d_feature)
                classifier_q = classifier_q.transpose(0, 1).transpose(1, 2)

                classifier_p = phi_pmu.unsqueeze(1).repeat(1, self.w_repeat, 1, 1).reshape(self.way_number, -1, self.d_feature)
                classifier_p = classifier_p.transpose(0, 1).transpose(1, 2)

            output_posterior = torch.bmm(repeat_predict_samples, classifier_q).unsqueeze(0)
            output_prior = torch.bmm(repeat_predict_samples, classifier_p).unsqueeze(0)
            output_posterior_list.append(output_posterior)
            output_prior_list.append(output_prior)

        kl_w = torch.cat(kl_w, 0)
        output_posterior_all = torch.cat(output_posterior_list, 0)
        output_prior_all = torch.cat(output_prior_list, 0)
        return kl_w, output_posterior_all, output_prior_all

    def forward(self, inputs_batch, labels_batch, a_repeat, w_repeat):
        '''
            inputs_batch 4 * 5 * 16 * 4096
            labels_batch 4 * 5 * 16 * 1
        '''

        # dividing the context and target samples---------------------------------------------------------
        label_all = labels_batch.squeeze(-1)
        _, indices = torch.sort(label_all[0, :, 0])

        x_c = inputs_batch[:, :, :self.shot_number, :]
        x_t = inputs_batch[:, :, self.shot_number:, :]

        x_c_order = x_c[:, indices, :, :]
        x_t_order = x_t[:, indices, :, :]

        # task-wise transformer--------------------------------------------------------------------------------
        kl_a, a_psample, a_qsample = self.task_wise_transformer_inference(x_c_order, x_t_order)
        kl_w, output_posterior_all, output_prior_all = self.class_wise_transformer_inference(a_qsample, a_psample, x_c_order, x_t_order, x_t)

        if self.training:
            return output_posterior_all, output_prior_all, kl_a, kl_w
        else:
            return output_prior_all


