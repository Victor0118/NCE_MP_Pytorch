from mp_cnn.trainers.sick_trainer import SICKTrainer
from mp_cnn.trainers.msrvid_trainer import MSRVIDTrainer
from mp_cnn.trainers.trecqa_trainer import TRECQATrainer
from mp_cnn.trainers.wikiqa_trainer import WikiQATrainer
from nce.NCE_MP_Pytorch.trainers.trecqa_trainer import TRECQATrainer_NCE
from nce.NCE_MP_Pytorch.trainers.wikiqa_trainer import WikiQATrainer_NCE


class MPCNNTrainerFactory(object):
    """
    Get the corresponding Trainer class for a particular dataset.
    """
    trainer_map = {
        'sick': SICKTrainer,
        'msrvid': MSRVIDTrainer,
        'trecqa': TRECQATrainer,
        'wikiqa': WikiQATrainer
    }

    trainer_map_nce = {
        'trecqa': TRECQATrainer_NCE,
        'wikiqa': WikiQATrainer_NCE
    }

    @staticmethod
    def get_trainer(dataset_name, model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator=None, nce=False):
        if nce:
            trainer_map = MPCNNTrainerFactory.trainer_map_nce
        else:
            trainer_map = MPCNNTrainerFactory.trainer_map

        if dataset_name not in trainer_map:
            raise ValueError('{} is not implemented.'.format(dataset_name))

        return trainer_map[dataset_name](
            model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator
        )
