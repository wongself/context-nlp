import numpy as np
import json
from loguru import logger
import random


def batchify(samples, batch_size):
    """
    Batchify samples with a batch size
    """
    num_samples = len(samples)

    list_samples_batches = []

    # if a sentence is too long, make itself a batch to avoid GPU OOM
    to_single_batch = []
    for i in range(0, len(samples)):
        if len(samples[i]['tokens']) > 350:
            to_single_batch.append(i)

    for i in to_single_batch:
        logger.info('Single batch sample: %s-%d', samples[i]['doc_key'],
                    samples[i]['sentence_ix'])
        list_samples_batches.append([samples[i]])
    samples = [
        sample for i, sample in enumerate(samples) if i not in to_single_batch
    ]

    for i in range(0, len(samples), batch_size):
        list_samples_batches.append(samples[i:i + batch_size])

    assert (sum([len(batch) for batch in list_samples_batches]) == num_samples)

    return list_samples_batches


def overlap(s1, s2):
    if s2.start_sent >= s1.start_sent and s2.start_sent <= s1.end_sent:
        return True
    if s2.end_sent >= s1.start_sent and s2.end_sent <= s1.end_sent:
        return True
    return False


def convert_dataset_to_samples(dataset,
                               args,
                               ner_label2id=None,
                               context_window=0,
                               split=0):
    """
    Extract sentences and gold entities from a dataset
    """
    # split: split the data into train and dev (for ACE04)
    # split == 0: don't split
    # split == 1: return first 90% (train)
    # split == 2: return last 10% (dev)
    samples = []
    num_ner = 0
    max_len = 0
    max_ner = 0
    num_overlap = 0

    if split == 0:
        data_range = (0, len(dataset))
    elif split == 1:
        data_range = (0, int(len(dataset) * 0.9))
    elif split == 2:
        data_range = (int(len(dataset) * 0.9), len(dataset))

    for c, doc in enumerate(dataset):
        if c < data_range[0] or c >= data_range[1]:
            continue
        for i, sent in enumerate(doc):
            sent_length = len(sent.text)
            num_ner += len(sent.ner)
            sample = {
                'doc_key': doc._doc_key,
                'sentence_ix': sent.sentence_ix,
            }
            if context_window != 0 and sent_length > context_window:
                logger.info(f'Long sentence: {sample} ({sent_length})')
                # print('Exclude:', sample)
                # continue
            sample['tokens'] = sent.text
            sample['sent_length'] = sent_length
            sent_start = 0
            sent_end = sent_length

            max_len = max(max_len, sent_length)
            max_ner = max(max_ner, len(sent.ner))

            if args.context_mode == 'both':
                add_left = (context_window - sent_length) // 2
                add_right = (context_window - sent_length) - add_left

                # add left context
                j = i - 1
                while j >= 0 and add_left > 0:
                    if args.truncate_sent and len(doc[j].text) > add_left:
                        break
                    context_to_add = doc[j].text[-add_left:]
                    sample['tokens'] = context_to_add + sample['tokens']
                    add_left -= len(context_to_add)
                    sent_start += len(context_to_add)
                    sent_end += len(context_to_add)
                    j -= 1

                # add right context
                j = i + 1
                while j < len(doc) and add_right > 0:
                    if args.truncate_sent and len(doc[j].text) > add_right:
                        break
                    context_to_add = doc[j].text[:add_right]
                    sample['tokens'] = sample['tokens'] + context_to_add
                    add_right -= len(context_to_add)
                    j += 1

            elif args.context_mode == 'left':
                add_left = context_window - sent_length

                # add left context
                j = i - 1
                while j >= 0 and add_left > 0:
                    if args.truncate_sent and len(doc[j].text) > add_left:
                        break
                    context_to_add = doc[j].text[-add_left:]
                    sample['tokens'] = context_to_add + sample['tokens']
                    add_left -= len(context_to_add)
                    sent_start += len(context_to_add)
                    sent_end += len(context_to_add)
                    j -= 1

            elif args.context_mode == 'right':
                add_right = context_window - sent_length

                # add right context
                j = i + 1
                while j < len(doc) and add_right > 0:
                    if args.truncate_sent and len(doc[j].text) > add_right:
                        break
                    context_to_add = doc[j].text[:add_right]
                    sample['tokens'] = sample['tokens'] + context_to_add
                    add_right -= len(context_to_add)
                    j += 1

            elif args.context_mode == 'random':
                add_right = context_window - sent_length

                # add random context
                while add_right > 0:
                    j = random.randrange(0, len(doc))
                    if args.truncate_sent and len(doc[j].text) > add_right:
                        break
                    context_to_add = doc[j].text[:add_right]
                    sample['tokens'] = sample['tokens'] + context_to_add
                    add_right -= len(context_to_add)

            sample['sent_start'] = sent_start
            sample['sent_end'] = sent_end
            sample['sent_start_in_doc'] = sent.sentence_start

            # create positive samples
            pos_spans = []
            sample['spans'] = []
            sample['spans_label'] = []
            for ner in sent.ner:
                i, j = ner.span.start_sent, ner.span.end_sent
                sample['spans'].append(
                    (i + sent_start, j + sent_start, j - i + 1))
                sample['spans_label'].append(ner_label2id[ner.label])
                pos_spans.append(ner.span.span_sent)

            # create negative samples
            neg_sample_spans = []
            for i in range(sent_length):
                for j in range(i, min(sent_length, i + args.max_span_length)):
                    if (i, j) in pos_spans:
                        continue
                    neg_sample_spans.append(
                        (i + sent_start, j + sent_start, j - i + 1))

            # combine pos/neg samples
            neg_sample_spans = random.sample(
                neg_sample_spans,
                min(len(neg_sample_spans), args.neg_entity_count))
            sample['spans'].extend(neg_sample_spans)
            sample['spans_label'].extend(
                [0 for _ in range(len(neg_sample_spans))])

            samples.append(sample)
    avg_length = sum([len(sample['tokens'])
                      for sample in samples]) / len(samples)
    max_length = max([len(sample['tokens']) for sample in samples])
    logger.info('# Overlap: %d' % num_overlap)  # TODO
    logger.info(
        f'Extracted {len(samples)} samples from {data_range[1] - data_range[0]} documents, '
        f'with {num_ner} NER labels, {avg_length:.3f} avg input length, {max_length} max length'
    )
    logger.info(f'Max Length: {max_len}, max NER: {max_ner}')
    return samples, num_ner


class NpEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def get_train_fold(data, fold):
    print('Getting train fold %d...' % fold)
    left = int(len(data) * 0.1 * fold)
    right = int(len(data) * 0.1 * (fold + 1))
    new_js = []
    new_docs = []
    for i in range(len(data)):
        if i < left or i >= right:
            new_js.append(data.js[i])
            new_docs.append(data.documents[i])
    print('# documents: %d --> %d' % (len(data), len(new_docs)))
    data.js = new_js
    data.documents = new_docs
    return data


def get_test_fold(data, fold):
    print('Getting test fold %d...' % fold)
    left = int(len(data) * 0.1 * fold)
    right = int(len(data) * 0.1 * (fold + 1))
    new_js = []
    new_docs = []
    for i in range(len(data)):
        if i >= left and i < right:
            new_js.append(data.js[i])
            new_docs.append(data.documents[i])
    print('# documents: %d --> %d' % (len(data), len(new_docs)))
    data.js = new_js
    data.documents = new_docs
    return data
