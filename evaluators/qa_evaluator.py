import numpy as np

import torch.nn.functional as F

from NCE_MP_Pytorch.evaluators.evaluator import Evaluator
from utils.relevancy_metrics import get_map_mrr


class QAEvaluator(Evaluator):

    def __init__(self, dataset_cls, model, data_loader, batch_size, device):
        super(QAEvaluator, self).__init__(dataset_cls, model, data_loader, batch_size, device)

    def get_scores(self):
        self.model.eval()
        qids = []
        predictions = []
        labels = []

        for batch in self.data_loader:
            '''
            # dev singlely or in a batch? -> in a batch
            but dev singlely is equal to dev_size = 1
            '''
            scores = self.model.convModel(batch.sentence_1, batch.sentence_2, batch.ext_feats)
            scores = self.model.linearLayer(scores)
            qid_array = np.transpose(batch.id.cpu().data.numpy())
            score_array = scores.cpu().data.numpy().reshape(-1)
            true_label_array = np.transpose(batch.label.cpu().data.numpy())

            qids.extend(qid_array.tolist())
            predictions.extend(score_array.tolist())
            labels.extend(true_label_array.tolist())

        # for batch in self.data_loader:
        #     qids.extend(batch.id.data.cpu().numpy())
        #     output = self.model(batch.sentence_1, batch.sentence_2, batch.ext_feats)
        #     test_cross_entropy_loss += F.cross_entropy(output, batch.label, size_average=False).data[0]
        #
        #     true_labels.extend(batch.label.data.cpu().numpy())
        #     predictions.extend(output.data.exp()[:, 1].cpu().numpy())
        #
        #     del output

        # qids = list(map(lambda n: int(round(n * 10, 0)) / 10, qids))

        mean_average_precision, mean_reciprocal_rank = get_map_mrr(qids, predictions, labels, self.data_loader.device)

        return [mean_average_precision, mean_reciprocal_rank], ['map', 'mrr']
