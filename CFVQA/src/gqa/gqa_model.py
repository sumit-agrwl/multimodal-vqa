# coding=utf-8
# Copyleft 2019 project LXRT.
import torch
import torch.nn as nn

from param import args
from coarse.lxrt.entry import LXRTEncoder
from coarse.lxrt.modeling import BertLayerNorm, GeLU
from fine.graphvqa.modeling import GraphVQAEncoder

# Max length including <bos> and <eos>
MAX_GQA_LENGTH = 20


class GQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        self.coarse_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_GQA_LENGTH
        )
        hid_dim = self.coarse_encoder.dim
        self.coarse_logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.coarse_logit_fc.apply(self.coarse_encoder.model.init_bert_weights)
        
        self.fine_encoder = GraphVQAEncoder(
            args
        )
        hid_dim = self.fine_encoder.question_hidden_dim  * 3 # due to concat
        out_classifier_dim = 512
        self.fine_logit_fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(hid_dim, out_classifier_dim),
            nn.ELU(),
            nn.Dropout(p=0.2),
            nn.Linear(out_classifier_dim, num_answers)
        )
        w_coarse = torch.empty(num_answers)
        w_fine = torch.empty(num_answers)
        nn.init.uniform_(w_coarse)
        nn.init.uniform_(w_fine)
        self.weight_coarse = nn.Parameter(w_coarse)
        self.weight_fine = nn.Parameter(w_fine)
        self.softmax = nn.Softmax(dim=0)
        self.num_answers = num_answers

    def forward(self, **kwargs):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        feat = kwargs['feats']
        pos = kwargs['boxes']
        ques = kwargs['ques']
        ques_fine = kwargs['ques_fine'] 
        sg = kwargs['sg']
        prog = kwargs['prog_in']
        val = False
        if 'val' in kwargs:
            val = kwargs['val']
        
        x = self.coarse_encoder(ques, (feat, pos))
        c_logit = self.coarse_logit_fc(x)
        if not val:
            y, prog_out, _, _ = self.fine_encoder(ques_fine, sg, prog, val)
        else:
            y, _, _ = self.fine_encoder(ques_fine, sg, prog, val)
        f_logit = self.fine_logit_fc(y)
        #weights = torch.vstack([self.weight_coarse, self.weight_fine])
        #w_coarse, w_fine = self.softmax(weights)
        #logit = (w_coarse * c_logit) + (w_fine * f_logit)
        logit = (0.5 * f_logit) + (0.5 * c_logit)
        if not val:
            return logit, prog_out
        return logit
