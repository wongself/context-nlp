import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from allennlp.nn.util import batched_index_select
from allennlp.modules import FeedForward

from transformers import BertTokenizer, BertPreTrainedModel, BertModel
from transformers import AlbertTokenizer, AlbertPreTrainedModel, AlbertModel
from transformers import RobertaTokenizer, RobertaPreTrainedModel, RobertaModel

import subprocess
import numpy
from loguru import logger


def auto_device(min_memory: int = 8192):
    logger.info("Automatically allocating device...")

    if not torch.cuda.is_available():
        logger.info("Cuda device is unavailable, device `cpu` returned")
        return torch.device('cpu')

    try:
        COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
        free_memories = subprocess.check_output(
            COMMAND.split()).decode().strip().split('\n')[1:]
        free_memories = [int(x.split()[0]) for x in free_memories]
        assert len(free_memories) == torch.cuda.device_count()
    except Exception:
        logger.warning(
            "Cuda device information inquiry failed, device `cpu` returned")
        return torch.device('cpu')
    else:
        selected_id = numpy.argmax(free_memories)
        selected_mem = free_memories[selected_id]
        if selected_mem < min_memory:
            logger.warning(
                f"Cuda device `cuda:{selected_id}` with maximum free memory {selected_mem} MiB "
                f"fails to meet the requirement {min_memory} MiB, device `cpu` returned"
            )
            return torch.device('cpu')
        else:
            logger.info(
                f"Cuda device `cuda:{selected_id}` with free memory {selected_mem} MiB "
                f"successfully allocated, device `cuda:{selected_id}` returned"
            )
            return torch.device('cuda', selected_id)


class BertForEntity(BertPreTrainedModel):

    def __init__(self,
                 config,
                 num_ner_labels,
                 head_hidden_dim=150,
                 width_embedding_dim=25):
        super().__init__(config)

        self.bert = BertModel(config)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.width_embedding = nn.Embedding(50, width_embedding_dim)

        self.ner_classifier = nn.Sequential(
            FeedForward(input_dim=config.hidden_size * 2 + width_embedding_dim,
                        num_layers=2,
                        hidden_dims=head_hidden_dim,
                        activations=nn.ReLU(),
                        dropout=0.2), nn.Linear(head_hidden_dim,
                                                num_ner_labels))

        self.init_weights()

    def _get_span_embeddings(self,
                             input_ids,
                             spans,
                             token_type_ids=None,
                             attention_mask=None):
        sequence_output, _ = self.bert(input_ids=input_ids,
                                       token_type_ids=token_type_ids,
                                       attention_mask=attention_mask,
                                       return_dict=False)

        sequence_output = self.hidden_dropout(sequence_output)
        """
        spans: [batch_size, num_spans, 3]; 0: left_ned, 1: right_end, 2: width
        spans_mask: (batch_size, num_spans, )
        """
        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_start_embedding = batched_index_select(sequence_output,
                                                     spans_start)
        spans_end = spans[:, :, 1].view(spans.size(0), -1)
        spans_end_embedding = batched_index_select(sequence_output, spans_end)

        spans_width = spans[:, :, 2].view(spans.size(0), -1)
        spans_width_embedding = self.width_embedding(spans_width)

        # Concatenate embeddings of left/right points and the width embedding
        spans_embedding = torch.cat(
            (spans_start_embedding, spans_end_embedding,
             spans_width_embedding),
            dim=-1)
        """
        spans_embedding: (batch_size, num_spans, hidden_size*2+embedding_dim)
        """
        return spans_embedding

    def forward(self,
                input_ids,
                spans,
                spans_mask,
                spans_ner_label=None,
                token_type_ids=None,
                attention_mask=None):
        spans_embedding = self._get_span_embeddings(
            input_ids,
            spans,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)
        ffnn_hidden = []
        hidden = spans_embedding
        for layer in self.ner_classifier:
            hidden = layer(hidden)
            ffnn_hidden.append(hidden)
        logits = ffnn_hidden[-1]

        if spans_ner_label is not None:
            loss_fct = CrossEntropyLoss(reduction='sum')
            if attention_mask is not None:
                active_loss = spans_mask.view(-1) == 1
                active_logits = logits.view(-1, logits.shape[-1])
                active_labels = torch.where(
                    active_loss, spans_ner_label.view(-1),
                    torch.tensor(
                        loss_fct.ignore_index).type_as(spans_ner_label))
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, logits.shape[-1]),
                                spans_ner_label.view(-1))
            return loss, logits, spans_embedding
        else:
            return logits, spans_embedding, spans_embedding


class DeBertaForEntity(RobertaPreTrainedModel):

    def __init__(self,
                 config,
                 num_ner_labels,
                 head_hidden_dim=150,
                 width_embedding_dim=25):
        super().__init__(config)

        self.bert = RobertaModel(config)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.width_embedding = nn.Embedding(50, width_embedding_dim)

        self.ner_classifier = nn.Sequential(
            FeedForward(input_dim=config.hidden_size * 2 + width_embedding_dim,
                        num_layers=2,
                        hidden_dims=head_hidden_dim,
                        activations=nn.ReLU(),
                        dropout=0.2), nn.Linear(head_hidden_dim,
                                                num_ner_labels))

        self.init_weights()

    def _get_span_embeddings(self,
                             input_ids,
                             spans,
                             token_type_ids=None,
                             attention_mask=None):
        sequence_output = self.bert(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)

        sequence_output = self.hidden_dropout(
            sequence_output.last_hidden_state)
        """
        spans: [batch_size, num_spans, 3]; 0: left_ned, 1: right_end, 2: width
        spans_mask: (batch_size, num_spans, )
        """
        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_start_embedding = batched_index_select(sequence_output,
                                                     spans_start)
        spans_end = spans[:, :, 1].view(spans.size(0), -1)
        spans_end_embedding = batched_index_select(sequence_output, spans_end)

        spans_width = spans[:, :, 2].view(spans.size(0), -1)
        spans_width_embedding = self.width_embedding(spans_width)

        # Concatenate embeddings of left/right points and the width embedding
        spans_embedding = torch.cat(
            (spans_start_embedding, spans_end_embedding,
             spans_width_embedding),
            dim=-1)
        """
        spans_embedding: (batch_size, num_spans, hidden_size*2+embedding_dim)
        """
        return spans_embedding

    def forward(self,
                input_ids,
                spans,
                spans_mask,
                spans_ner_label=None,
                token_type_ids=None,
                attention_mask=None):
        spans_embedding = self._get_span_embeddings(
            input_ids,
            spans,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)
        ffnn_hidden = []
        hidden = spans_embedding
        for layer in self.ner_classifier:
            hidden = layer(hidden)
            ffnn_hidden.append(hidden)
        logits = ffnn_hidden[-1]

        if spans_ner_label is not None:
            loss_fct = CrossEntropyLoss(reduction='sum')
            if attention_mask is not None:
                active_loss = spans_mask.view(-1) == 1
                active_logits = logits.view(-1, logits.shape[-1])
                active_labels = torch.where(
                    active_loss, spans_ner_label.view(-1),
                    torch.tensor(
                        loss_fct.ignore_index).type_as(spans_ner_label))
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, logits.shape[-1]),
                                spans_ner_label.view(-1))
            return loss, logits, spans_embedding
        else:
            return logits, spans_embedding, spans_embedding


class AlbertForEntity(AlbertPreTrainedModel):

    def __init__(self,
                 config,
                 num_ner_labels,
                 head_hidden_dim=150,
                 width_embedding_dim=25):
        super().__init__(config)

        self.albert = AlbertModel(config)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.width_embedding = nn.Embedding(50, width_embedding_dim)

        self.ner_classifier = nn.Sequential(
            FeedForward(input_dim=config.hidden_size * 2 + width_embedding_dim,
                        num_layers=2,
                        hidden_dims=head_hidden_dim,
                        activations=nn.ReLU(),
                        dropout=0.2), nn.Linear(head_hidden_dim,
                                                num_ner_labels))

        self.init_weights()

    def _get_span_embeddings(self,
                             input_ids,
                             spans,
                             token_type_ids=None,
                             attention_mask=None):
        sequence_output, _ = self.albert(input_ids=input_ids,
                                         token_type_ids=token_type_ids,
                                         attention_mask=attention_mask,
                                         return_dict=False)

        sequence_output = self.hidden_dropout(sequence_output)
        """
        spans: [batch_size, num_spans, 3]; 0: left_ned, 1: right_end, 2: width
        spans_mask: (batch_size, num_spans, )
        """
        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_start_embedding = batched_index_select(sequence_output,
                                                     spans_start)
        spans_end = spans[:, :, 1].view(spans.size(0), -1)
        spans_end_embedding = batched_index_select(sequence_output, spans_end)

        spans_width = spans[:, :, 2].view(spans.size(0), -1)
        spans_width_embedding = self.width_embedding(spans_width)

        spans_embedding = torch.cat(
            (spans_start_embedding, spans_end_embedding,
             spans_width_embedding),
            dim=-1)
        """
        spans_embedding: (batch_size, num_spans, hidden_size*2+embedding_dim)
        """
        return spans_embedding

    def forward(self,
                input_ids,
                spans,
                spans_mask,
                spans_ner_label=None,
                token_type_ids=None,
                attention_mask=None):
        spans_embedding = self._get_span_embeddings(
            input_ids,
            spans,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)
        ffnn_hidden = []
        hidden = spans_embedding
        for layer in self.ner_classifier:
            hidden = layer(hidden)
            ffnn_hidden.append(hidden)
        logits = ffnn_hidden[-1]

        if spans_ner_label is not None:
            loss_fct = CrossEntropyLoss(reduction='sum')
            if attention_mask is not None:
                active_loss = spans_mask.view(-1) == 1
                active_logits = logits.view(-1, logits.shape[-1])
                active_labels = torch.where(
                    active_loss, spans_ner_label.view(-1),
                    torch.tensor(
                        loss_fct.ignore_index).type_as(spans_ner_label))
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, logits.shape[-1]),
                                spans_ner_label.view(-1))
            return loss, logits, spans_embedding
        else:
            return logits, spans_embedding, spans_embedding


class EntityModel():

    def __init__(self, args, num_ner_labels):
        super().__init__()

        model_path = args.model_dir
        logger.info(f'Load model from {model_path}')

        if model_path.lower().find('albert') >= 0:
            logger.info('Loading AlBert...')
            self.tokenizer = AlbertTokenizer.from_pretrained(model_path)
            self.bert_model = AlbertForEntity.from_pretrained(
                model_path, num_ner_labels=num_ner_labels)
        elif model_path.lower().find('roberta') >= 0:
            logger.info('Loading RoBerta...')
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
            self.bert_model = DeBertaForEntity.from_pretrained(
                model_path, num_ner_labels=num_ner_labels)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.bert_model = BertForEntity.from_pretrained(
                model_path, num_ner_labels=num_ner_labels)

        device = auto_device() if args.gpu_id < 0 else torch.device(
            'cuda', args.gpu_id)
        if device.type.startswith('cuda'):
            torch.cuda.set_device(device)
            # temp = torch.randn(50).to(device)

        self._device = device
        self.bert_model.to(device)

    def _get_input_tensors(self, tokens, spans, spans_ner_label):
        start2idx = []
        end2idx = []

        bert_tokens = []
        bert_tokens.append(self.tokenizer.cls_token)
        for token in tokens:
            start2idx.append(len(bert_tokens))
            sub_tokens = self.tokenizer.tokenize(token)
            bert_tokens += sub_tokens
            end2idx.append(len(bert_tokens) - 1)
        bert_tokens.append(self.tokenizer.sep_token)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])

        bert_spans = [[start2idx[span[0]], end2idx[span[1]], span[2]]
                      for span in spans]
        bert_spans_tensor = torch.tensor([bert_spans])

        spans_ner_label_tensor = torch.tensor([spans_ner_label])

        return tokens_tensor, bert_spans_tensor, spans_ner_label_tensor

    def _get_input_tensors_batch(self, samples_list, training=True):
        tokens_tensor_list = []
        bert_spans_tensor_list = []
        spans_ner_label_tensor_list = []
        sentence_length = []

        max_tokens = 0
        max_spans = 0
        for sample in samples_list:
            tokens = sample['tokens']
            spans = sample['spans']
            spans_ner_label = sample['spans_label']

            tokens_tensor, bert_spans_tensor, spans_ner_label_tensor = self._get_input_tensors(
                tokens, spans, spans_ner_label)
            tokens_tensor_list.append(tokens_tensor)
            bert_spans_tensor_list.append(bert_spans_tensor)
            spans_ner_label_tensor_list.append(spans_ner_label_tensor)
            assert (
                bert_spans_tensor.shape[1] == spans_ner_label_tensor.shape[1])
            if (tokens_tensor.shape[1] > max_tokens):
                max_tokens = tokens_tensor.shape[1]
            if (bert_spans_tensor.shape[1] > max_spans):
                max_spans = bert_spans_tensor.shape[1]
            sentence_length.append(sample['sent_length'])
        sentence_length = torch.Tensor(sentence_length)

        # apply padding and concatenate tensors
        final_tokens_tensor = None
        final_attention_mask = None
        final_bert_spans_tensor = None
        final_spans_ner_label_tensor = None
        final_spans_mask_tensor = None
        for tokens_tensor, bert_spans_tensor, spans_ner_label_tensor in zip(
                tokens_tensor_list, bert_spans_tensor_list,
                spans_ner_label_tensor_list):
            # padding for tokens
            num_tokens = tokens_tensor.shape[1]
            tokens_pad_length = max_tokens - num_tokens
            attention_tensor = torch.full([1, num_tokens], 1, dtype=torch.long)
            if tokens_pad_length > 0:
                pad = torch.full([1, tokens_pad_length],
                                 self.tokenizer.pad_token_id,
                                 dtype=torch.long)
                tokens_tensor = torch.cat((tokens_tensor, pad), dim=1)
                attention_pad = torch.full([1, tokens_pad_length],
                                           0,
                                           dtype=torch.long)
                attention_tensor = torch.cat((attention_tensor, attention_pad),
                                             dim=1)

            # padding for spans
            num_spans = bert_spans_tensor.shape[1]
            spans_pad_length = max_spans - num_spans
            spans_mask_tensor = torch.full([1, num_spans], 1, dtype=torch.long)
            if spans_pad_length > 0:
                pad = torch.full(
                    [1, spans_pad_length, bert_spans_tensor.shape[2]],
                    0,
                    dtype=torch.long)
                bert_spans_tensor = torch.cat((bert_spans_tensor, pad), dim=1)
                mask_pad = torch.full([1, spans_pad_length],
                                      0,
                                      dtype=torch.long)
                spans_mask_tensor = torch.cat((spans_mask_tensor, mask_pad),
                                              dim=1)
                spans_ner_label_tensor = torch.cat(
                    (spans_ner_label_tensor, mask_pad), dim=1)

            # update final outputs
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor
                final_attention_mask = attention_tensor
                final_bert_spans_tensor = bert_spans_tensor
                final_spans_ner_label_tensor = spans_ner_label_tensor
                final_spans_mask_tensor = spans_mask_tensor
            else:
                final_tokens_tensor = torch.cat(
                    (final_tokens_tensor, tokens_tensor), dim=0)
                final_attention_mask = torch.cat(
                    (final_attention_mask, attention_tensor), dim=0)
                final_bert_spans_tensor = torch.cat(
                    (final_bert_spans_tensor, bert_spans_tensor), dim=0)
                final_spans_ner_label_tensor = torch.cat(
                    (final_spans_ner_label_tensor, spans_ner_label_tensor),
                    dim=0)
                final_spans_mask_tensor = torch.cat(
                    (final_spans_mask_tensor, spans_mask_tensor), dim=0)

        return final_tokens_tensor, final_attention_mask, final_bert_spans_tensor, final_spans_mask_tensor, final_spans_ner_label_tensor, sentence_length

    def run_batch(self, samples_list, try_cuda=True, training=True):
        # convert samples to input tensors
        tokens_tensor, attention_mask_tensor, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, sentence_length = self._get_input_tensors_batch(
            samples_list, training)

        output_dict = {
            'ner_loss': 0,
        }

        if training:
            self.bert_model.train()
            ner_loss, ner_logits, spans_embedding = self.bert_model(
                input_ids=tokens_tensor.to(self._device),
                spans=bert_spans_tensor.to(self._device),
                spans_mask=spans_mask_tensor.to(self._device),
                spans_ner_label=spans_ner_label_tensor.to(self._device),
                attention_mask=attention_mask_tensor.to(self._device),
            )
            output_dict['ner_loss'] = ner_loss.sum()
            output_dict['ner_llh'] = F.log_softmax(ner_logits, dim=-1)
        else:
            self.bert_model.eval()
            with torch.no_grad():
                ner_logits, spans_embedding, last_hidden = self.bert_model(
                    input_ids=tokens_tensor.to(self._device),
                    spans=bert_spans_tensor.to(self._device),
                    spans_mask=spans_mask_tensor.to(self._device),
                    spans_ner_label=None,
                    attention_mask=attention_mask_tensor.to(self._device),
                )
            _, predicted_label = ner_logits.max(2)
            predicted_label = predicted_label.cpu().numpy()
            last_hidden = last_hidden.cpu().numpy()

            predicted = []
            pred_prob = []
            hidden = []
            for i, sample in enumerate(samples_list):
                ner = []
                prob = []
                lh = []
                for j in range(len(sample['spans'])):
                    ner.append(predicted_label[i][j])
                    # prob.append(F.softmax(ner_logits[i][j], dim=-1).cpu().numpy())
                    prob.append(ner_logits[i][j].cpu().numpy())
                    lh.append(last_hidden[i][j])
                predicted.append(ner)
                pred_prob.append(prob)
                hidden.append(lh)
            output_dict['pred_ner'] = predicted
            output_dict['ner_probs'] = pred_prob
            output_dict['ner_last_hidden'] = hidden

        return output_dict
