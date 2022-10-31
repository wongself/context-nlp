from typing import List
import json

error_cnt = 0


def load_documents(dataset_file):
    documents = []
    sentences = []  # all words
    words = []
    ner = []  # all labels
    labels = []
    count = 0
    start = 0

    with open(dataset_file) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('-DOCSTART-'):
                if len(sentences) > 0:
                    assert len(sentences) == len(ner)
                    documents.append(
                        dict(sentences=sentences,
                             ner=ner,
                             doc_key=f'conll03-{count}'))
                    count += 1
                    start = 0
                    sentences, ner = [], []
                continue

            if not line:
                if len(words) > 0:
                    assert len(words) == len(labels)
                    sentences.append(words)
                    ner.append(bio_tags_to_spans(labels, start))
                    start += len(words)
                    words, labels = [], []
            else:
                items = line.split(' ')
                words.append(items[0])
                labels.append(items[-1])

    if len(sentences) > 0:
        assert len(sentences) == len(ner)
        documents.append(
            dict(sentences=sentences, ner=ner, doc_key=f'conll03-{count}'))

    return documents


def bio_tags_to_spans(tag_sequence: List[str], start_idx: int = 0):
    spans = []
    span_start = 0
    span_end = 0
    active_tag = None
    for idx, str_tag in enumerate(tag_sequence):
        bio_tag = str_tag[0]
        assert bio_tag in ["B", "I", "O"]
        conll_tag = str_tag[2:]
        if bio_tag == "O":
            if active_tag is not None:
                spans.append([span_start, span_end, active_tag])
            active_tag = None
        elif bio_tag == "B":
            if active_tag is not None:
                spans.append([span_start, span_end, active_tag])
            active_tag = conll_tag
            span_start = idx
            span_end = idx
        elif bio_tag == "I" and conll_tag == active_tag:
            span_end += 1
        else:
            print('---error---')
            if active_tag is not None:
                spans.append([span_start, span_end, active_tag])
            active_tag = conll_tag
            span_start = idx
            span_end = idx
    if active_tag is not None:
        spans.append([span_start, span_end, active_tag])
    return spans


def bio_tags_to_spans(tag_sequence: List[str], start_idx: int = 0):
    global error_cnt
    spans = []
    span_start = 0
    span_end = 0
    active_tag = None
    for idx, str_tag in enumerate(tag_sequence):
        # Actual BIO tag.
        bio_tag = str_tag[0]
        assert bio_tag in ["B", "I", "O"]
        conll_tag = str_tag[2:]
        if bio_tag == "O":
            if active_tag is not None:
                spans.append([span_start, span_end, active_tag])
            active_tag = None
        elif bio_tag == "B":
            if active_tag is not None:
                spans.append([span_start, span_end, active_tag])
            active_tag = conll_tag
            span_start = idx + start_idx
            span_end = idx + start_idx
        elif bio_tag == "I" and conll_tag == active_tag:
            span_end += 1
        else:
            error_cnt += 1
            # if error_cnt < 5:
            #     print(tag_sequence, str_tag)
            if active_tag is not None:
                spans.append([span_start, span_end, active_tag])
            active_tag = conll_tag
            span_start = idx + start_idx
            span_end = idx + start_idx
    # Last token might have been a part of a valid span.
    if active_tag is not None:
        spans.append([span_start, span_end, active_tag])
    return spans


if __name__ == '__main__':
    path = 'train'
    docs = load_documents(f'./source/{path}.txt')
    # print(docs)
    print(error_cnt)

    with open(f'./output/{path}.json', 'w') as f:
        for doc in docs:
            f.write(json.dumps(doc) + '\n')
