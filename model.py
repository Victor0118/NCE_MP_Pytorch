import torch
import torch.nn as nn

from mp_cnn.model import MPCNN

class PairwiseConv(nn.Module):
    """Pairwise model based on MP-CNN"""
    def __init__(self, model):
        super(PairwiseConv, self).__init__()
        self.convModel = model
        self.dropout = nn.Dropout(model.dropout)
        self.linearLayer = nn.Linear(model.n_hidden, 1)
        self.posModel = self.convModel
        self.negModel = self.convModel

    def forward(self, input):
        pos = self.posModel(input[0].sentence_1, input[0].sentence_2, input[0].ext_feats)
        neg = self.negModel(input[1].sentence_1, input[1].sentence_2, input[1].ext_feats)
        pos = self.dropout(pos)
        neg = self.dropout(neg)
        pos = self.linearLayer(pos)
        neg = self.linearLayer(neg)
        combine = torch.cat([pos, neg], 1)
        return combine

class MPCNN4NCE(MPCNN):

        def __init__(self, embedding, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units,
                     num_classes, dropout, ext_feats):
            super(MPCNN4NCE, self).__init__(embedding, n_holistic_filters, n_per_dim_filters, filter_widths,
                                            hidden_layer_units, num_classes, dropout, ext_feats)
            self.dropout = dropout
            self.n_hidden = hidden_layer_units
            self.final_layers = nn.Sequential(
                nn.Linear(self.n_feat, hidden_layer_units),
                nn.Tanh()
            )
