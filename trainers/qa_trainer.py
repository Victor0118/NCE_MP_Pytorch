import time
import random
import heapq
import operator

import torch
import torch.nn.functional as F
from torchtext import data
from torch.optim.lr_scheduler import ReduceLROnPlateau

from NCE_MP_Pytorch.trainers.trainer import Trainer


class QATrainer(Trainer):

    def __init__(self, model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator=None):
        super(QATrainer, self).__init__(model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator)
        self.loss = torch.nn.MarginRankingLoss(margin=1, size_average=True)
        self.question2answer = {}
        self.best_dev_map = 0
        self.best_dev_mrr = 0
        self.false_samples = {}
        self.question2answer = {}
        self.early_stop = False
        self.start = time.time()
        self.iters_not_improved = 0
        self.q2neg = {}
        self.dev_log_template = ' '.join(
            '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>11.6f},{:>11.6f},{:12.6f},{:8.4f}'.split(','))
        self.log_template = ' '.join(
            '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>11.6f},{:>11.6f},'.split(','))

    # get the nearest negative samples to the positive sample by computing the feature difference
    def get_nearest_neg_id(self, pos_feature, neg_dict, distance="cosine", k=1):
        dis_list = []
        pos_feature = pos_feature.data.cpu().numpy()
        pos_feature_norm = pos_feature / np.sqrt(sum(pos_feature ** 2))
        neg_list = []
        for key in neg_dict:
            if distance == "l2":
                dis = np.sqrt(np.sum((np.array(pos_feature) - neg_dict[key]["feature"]) ** 2))
            elif distance == "cosine":
                neg_feature = np.array(neg_dict[key]["feature"])
                feat_norm = neg_feature / np.sqrt(sum(neg_feature ** 2))
                dis = 1 - feat_norm.dot(pos_feature_norm)
            dis_list.append(dis)
            neg_list.append(key)

        k = min(k, len(neg_dict))
        min_list = heapq.nsmallest(k, enumerate(dis_list), key=operator.itemgetter(1))
        min_id_list = [neg_list[x[0]] for x in min_list]
        return min_id_list

    # get the negative samples randomly
    def get_random_neg_id(self, q2neg, qid_i, k=5):
        # question 1734 has no neg answer
        if qid_i not in q2neg:
            return []
        k = min(k, len(q2neg[qid_i]))
        ran = random.sample(q2neg[qid_i], k)
        return ran

    # pack the lists of question/answer/ext_feat into a torchtext batch
    def get_batch(self, question, answer, ext_feat, size):
        new_batch = data.Batch()
        new_batch.batch_size = size
        setattr(new_batch, "sentence_2", torch.stack(answer))
        setattr(new_batch, "sentence_1", torch.stack(question))
        setattr(new_batch, "ext_feats", torch.stack(ext_feat))
        return new_batch

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        acc = 0
        tot = 0
        for batch_idx, batch in enumerate(self.train_loader):

            features = self.model.convModel(batch.sentence_1, batch.sentence_2, batch.ext_feats)
            new_train_pos = {"answer": [], "question": [], "ext_feat": []}
            new_train_neg = {"answer": [], "question": [], "ext_feat": []}
            max_len_q = 0
            max_len_a = 0

            batch_near_list = []
            batch_qid = []
            batch_aid = []

            for i in range(batch.batch_size):
                label_i = batch.label[i].cpu().data.numpy()[0]
                question_i = batch.sentence_1[i]
                # question_i = question_i[question_i!=1] # remove padding 1 <pad>
                answer_i = batch.sentence_2[i]
                # answer_i = answer_i[answer_i!=1] # remove padding 1 <pad>
                ext_feat_i = batch.ext_feats[i]
                qid_i = batch.id[i].data.cpu().numpy()[0]
                aid_i = batch.aid[i].data.cpu().numpy()[0]

                if qid_i not in self.question2answer:
                    self.question2answer[qid_i] = {"question": question_i, "pos": {}, "neg": {}}
                if label_i == 1:

                    if aid_i not in self.question2answer[qid_i]["pos"]:
                        self.question2answer[qid_i]["pos"][aid_i] = {}

                    self.question2answer[qid_i]["pos"][aid_i]["answer"] = answer_i
                    self.question2answer[qid_i]["pos"][aid_i]["ext_feat"] = ext_feat_i

                    # get neg samples in the first epoch but do not train
                    if epoch == 1:
                        continue
                    # random generate sample in the first training epoch
                    elif epoch == 2 or self.neg_sample == "random":
                        near_list = self.get_random_neg_id(self.q2neg, qid_i, k=self.neg_num)
                    else:
                        # debug_qid = qid_i
                        near_list = self.get_nearest_neg_id(features[i], self.question2answer[qid_i]["neg"], distance="cosine", k=self.neg_num)

                    batch_near_list.extend(near_list)

                    neg_size = len(near_list)
                    if neg_size != 0:
                        answer_i = answer_i[answer_i != 1]  # remove padding 1 <pad>
                        question_i = question_i[question_i != 1]  # remove padding 1 <pad>
                        for near_id in near_list:
                            batch_qid.append(qid_i)
                            batch_aid.append(aid_i)

                            new_train_pos["answer"].append(answer_i)
                            new_train_pos["question"].append(question_i)
                            new_train_pos["ext_feat"].append(ext_feat_i)

                            near_answer = self.question2answer[qid_i]["neg"][near_id]["answer"]
                            if question_i.size()[0] > max_len_q:
                                max_len_q = question_i.size()[0]
                            if near_answer.size()[0] > max_len_a:
                                max_len_a = near_answer.size()[0]
                            if answer_i.size()[0] > max_len_a:
                                max_len_a = answer_i.size()[0]

                            ext_feat_neg = self.question2answer[qid_i]["neg"][near_id]["ext_feat"]
                            new_train_neg["answer"].append(near_answer)
                            new_train_neg["question"].append(question_i)
                            new_train_neg["ext_feat"].append(ext_feat_neg)

                elif label_i == 0:

                    if aid_i not in self.question2answer[qid_i]["neg"]:
                        answer_i = answer_i[answer_i != 1]
                        self.question2answer[qid_i]["neg"][aid_i] = {"answer": answer_i}

                    self.question2answer[qid_i]["neg"][aid_i]["feature"] = features[i].data.cpu().numpy()
                    self.question2answer[qid_i]["neg"][aid_i]["ext_feat"] = ext_feat_i

                    if epoch == 1:
                        if qid_i not in self.q2neg:
                            self.q2neg[qid_i] = []

                        self.q2neg[qid_i].append(aid_i)

                        # pack the selected pos and neg samples into the torchtext batch and train
            if epoch != 1:
                true_batch_size = len(new_train_neg["answer"])
                if true_batch_size != 0:
                    for j in range(true_batch_size):
                        new_train_neg["answer"][j] = F.pad(new_train_neg["answer"][j],
                                                           (0, max_len_a - new_train_neg["answer"][j].size()[0]),
                                                           value=1)
                        new_train_pos["answer"][j] = F.pad(new_train_pos["answer"][j],
                                                           (0, max_len_a - new_train_pos["answer"][j].size()[0]),
                                                           value=1)
                        new_train_pos["question"][j] = F.pad(new_train_pos["question"][j],
                                                             (0, max_len_q - new_train_pos["question"][j].size()[0]),
                                                             value=1)
                        new_train_neg["question"][j] = F.pad(new_train_neg["question"][j],
                                                             (0, max_len_q - new_train_neg["question"][j].size()[0]),
                                                             value=1)

                    pos_batch = self.get_batch(new_train_pos["question"], new_train_pos["answer"], new_train_pos["ext_feat"],
                                          true_batch_size)
                    neg_batch = self.get_batch(new_train_neg["question"], new_train_neg["answer"], new_train_neg["ext_feat"],
                                          true_batch_size)

                    self.optimizer.zero_grad()
                    output = self.model([pos_batch, neg_batch])

                    cmp = output[:, 0] > output[:, 1]
                    acc += sum(cmp.data.cpu().numpy())
                    tot += true_batch_size

                    loss = self.loss(output[:, 0], output[:, 1], torch.autograd.Variable(torch.ones(1).cuda(device=self.device_id)))
                    loss_num = loss.data.cpu().numpy()[0]
                    loss.backward()
                    self.optimizer.step()
                    # Evaluate performance on validation set
                    if 0 % self.dev_log_interval == 1 and epoch != 1:
                        # switch model into evaluation mode

                        dev_scores = self.evaluate(self.dev_evaluator, 'dev')
                        dev_map, dev_mrr = dev_scores
                        print(self.dev_log_template.format(time.time() - self.start,
                                                      epoch, batch_idx, 0, 0, 0,
                                                      loss_num, acc / tot, dev_map, dev_mrr))

                        if self.best_dev_mrr < self.dev_mrr:
                            torch.save(self.model, self.model_outfile)
                            self.iters_not_improved = 0
                            self.best_dev_mrr = self.dev_mrr
                        else:
                            self.iters_not_improved += 1
                            if self.iters_not_improved >= self.patience:
                                self.early_stop = True
                                break


                    if batch_idx % self.log_interval == 1 and epoch != 1:
                        # logger.info progress message
                        self.logger.info(self.log_template.format(time.time() - self.start,
                                                  epoch, 0, 1 + batch_idx, 0, 0,
                                                  loss_num, acc / tot))

            # self.optimizer.zero_grad()
            # output = self.model(batch.sentence_1, batch.sentence_2, batch.ext_feats)
            # loss = F.cross_entropy(output, batch.label, size_average=False)
            # total_loss += loss.data[0]
            # loss.backward()
            # self.optimizer.step()
            # if batch_idx % self.log_interval == 0:
            #     self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, min(batch_idx * self.batch_size, len(batch.dataset.examples)),
            #         len(batch.dataset.examples),
            #         100. * batch_idx / (len(self.train_loader)), loss.data[0])
            #     )

        average_loss, train_map, train_mrr = self.evaluate(self.train_evaluator, 'train')

        if self.use_tensorboard:
            self.writer.add_scalar('{}/train/cross_entropy_loss'.format(self.train_loader.dataset.NAME), average_loss, epoch)
            self.writer.add_scalar('{}/train/map'.format(self.train_loader.dataset.NAME), train_map, epoch)
            self.writer.add_scalar('{}/train/mrr'.format(self.train_loader.dataset.NAME), train_mrr, epoch)

        return total_loss

    def train(self, epochs):
        scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=self.lr_reduce_factor, patience=self.patience)
        epoch_times = []
        best_dev_score = -1
        self.start = time.time()
        for epoch in range(1, epochs + 1):
            start = time.time()
            self.logger.info('Epoch {} started...'.format(epoch))
            self.train_epoch(epoch)

            dev_scores = self.evaluate(self.dev_evaluator, 'dev')
            dev_map, dev_mrr = dev_scores

            if self.use_tensorboard:
                self.writer.add_scalar('{}/lr'.format(self.train_loader.dataset.NAME), self.optimizer.param_groups[0]['lr'], epoch)
                self.writer.add_scalar('{}/dev/cross_entropy_loss'.format(self.train_loader.dataset.NAME), new_loss, epoch)
                self.writer.add_scalar('{}/dev/map'.format(self.train_loader.dataset.NAME), dev_map, epoch)
                self.writer.add_scalar('{}/dev/mrr'.format(self.train_loader.dataset.NAME), dev_mrr, epoch)

            end = time.time()
            duration = end - start
            self.logger.info('Epoch {} finished in {:.2f} minutes'.format(epoch, duration / 60))
            epoch_times.append(duration)

            if self.early_stop:
                self.logger.log("Early Stopping. Epoch: {}, Best Dev Loss: {}".format(epoch, best_dev_loss))
                break

            self.logger.info(self.dev_log_template.format(time.time() - start,
                                                          epoch, 0, 0, 0, 0, 0, 0, dev_map, dev_mrr))
            if self.best_dev_mrr < dev_mrr:
                torch.save(self.model, self.model_outfile)
                self.iters_not_improved = 0
                self.best_dev_mrr = dev_mrr
            else:
                self.iters_not_improved += 1
                if self.iters_not_improved >= self.patience:
                    self.early_stop = True
                    break

            if dev_scores[0] > best_dev_score:
                best_dev_score = dev_scores[0]
                torch.save(self.model, self.model_outfile)

            scheduler.step(dev_scores[0])

        self.logger.info('Training took {:.2f} minutes overall...'.format(sum(epoch_times) / 60))
