import math

from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from fairseq import metrics, utils

from collections import deque

import torch
import torch.nn as nn


@register_criterion("label_smoothed_cross_entropy_with_contrastive")
class LabelSmoothedCrossEntropyCriterionWithContrastive(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, sentence_avg, label_smoothing, ignore_prefix_size=0, report_accuracy=False,
                 contrastive_lambda=0.0,
                 temperature=1.0):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.contrastive_lambda = contrastive_lambda
        self.temperature = temperature
    
    @staticmethod
    def add_args(parser):
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument("--contrastive-lambda", type=float,
                            default=0.0,
                            help="The contrastive loss weight")
        parser.add_argument("--temperature", type=float,
                            default=1.0,)
    
    def swap_sample(self, sample):
        target = sample["target"]
        prev_output_tokens = sample["net_input"]["prev_output_tokens"]
        src_tokens = torch.cat((prev_output_tokens[:, :1], sample["net_input"]['src_tokens']), dim=-1)
        return {
            "net_input": {
                "src_tokens": target.contiguous(),
                "src_lengths": (target != self.padding_idx).int().sum(dim=1),
                "prev_output_tokens": src_tokens[:, :-1].contiguous()
            },
            'nsentences': sample['nsentences'],
            'ntokens': utils.item((src_tokens[:, 1:] != self.padding_idx).int().sum().data),
            "target": src_tokens[:, 1:].contiguous(),
            "id": sample["id"],
        }
    
    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        encoder_out = model.encoder.forward(sample["net_input"]["src_tokens"], sample["net_input"]["src_lengths"]).encoder_out
        reverse_sample = self.swap_sample(sample)
        reversed_encoder_out = model.encoder.forward(reverse_sample["net_input"]["src_tokens"], reverse_sample["net_input"]["src_lengths"]).encoder_out
        contrastive_loss = self.get_contrastive_loss(
            encoder_out,
            reversed_encoder_out,
            sample,
            reverse_sample,
        )
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        nsentences = sample["target"].size(0)
        ntokens = sample["ntokens"]
        all_loss = loss + contrastive_loss * self.contrastive_lambda * ntokens / nsentences
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        if isinstance(contrastive_loss, int):
            logging_output["contrastive_loss"] = 0
        else:
            logging_output["contrastive_loss"] = utils.item(contrastive_loss.data)
        
        return all_loss, sample_size, logging_output
    
    def similarity_function(self, ):
        return nn.CosineSimilarity(dim=-1)
    
    def get_contrastive_loss(self, encoder_out1, encoder_out2, sample1, sample2):
        
        def _sentence_embedding(encoder_out, sample):
            encoder_output = encoder_out.transpose(0, 1)
            src_tokens = sample["net_input"]["src_tokens"]
            mask = (src_tokens != self.padding_idx)
            encoder_embedding = (encoder_output * mask.unsqueeze(-1)).sum(dim=1) / mask.float().sum(dim=1).unsqueeze(-1)  # [batch, hidden_size]
            return encoder_embedding
        
        encoder_embedding1 = _sentence_embedding(encoder_out1, sample1)  # [batch, hidden_size]
        encoder_embedding2 = _sentence_embedding(encoder_out2, sample2)  # [batch, hidden_size]
        
        batch_size = encoder_embedding2.shape[0]
        feature_dim = encoder_embedding2.shape[1]
        anchor_feature = encoder_embedding1
        contrast_feature = encoder_embedding2
        
        similarity_function = self.similarity_function()
        anchor_dot_contrast = similarity_function(anchor_feature.expand((batch_size, batch_size, feature_dim)),
                                                  torch.transpose(contrast_feature.expand((batch_size, batch_size, feature_dim)), 0, 1))
        
        loss = -nn.LogSoftmax(0)(torch.div(anchor_dot_contrast, self.temperature)).diag().sum()
        
        return loss
    
    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        contrastive_loss = utils.item(
            sum(log.get("contrastive_loss", 0) for log in logging_outputs)
        )
        metrics.log_scalar(
            "contrastive_loss",
            contrastive_loss / nsentences / math.log(2),
            nsentences,
            round=3,
        )
