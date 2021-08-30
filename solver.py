import os
import torch

from models.bilstm_crf import BiLSTM_CRF_Runner
from utils.metrics import F1_PR

import logging
logger=logging.getLogger()


class NER_BiLSTM_CRF_Solver:
    def __init__(self,args):
        self.args=args

    def train_and_eval(self,train_data, dev_data, test_data,
                          word2id, tag2id, crf=True, remove_O=False):
        train_word_lists, train_tag_lists = train_data
        dev_word_lists, dev_tag_lists = dev_data
        test_word_lists, test_tag_lists = test_data

        vocab_size = len(word2id)
        out_size = len(tag2id)
        self.model = BiLSTM_CRF_Runner(self.args,vocab_size, out_size, crf=crf)
        trian_metrics,val_metrics=self.model.train(train_word_lists, train_tag_lists,
                        dev_word_lists, dev_tag_lists, word2id, tag2id)

        

        self.model_name = "bilstm_crf" if crf else "bilstm"
        
        logger.info("评估{}模型中...".format(self.model_name))
        self.save_model()
        pred_tag_lists, test_tag_lists = self.model.test(
            test_word_lists, test_tag_lists, word2id, tag2id)

        metrics = F1_PR(test_tag_lists, pred_tag_lists, remove_O=remove_O)
        metrics.report_scores()
        metrics.report_confusion_matrix()

        return trian_metrics,val_metrics

    def save_model(self,checkpoint_dir=None):
        ckpt_dir=self.args.checkpoint_dir
        if checkpoint_dir is not None:
            ckpt_dir=checkpoint_dir
        saved_path = os.path.join(ckpt_dir, self.model_name+".pth")
        torch.save({
            "model": self.model,
        }, saved_path)
        logger.info("Model has been saved to %s" % saved_path)      