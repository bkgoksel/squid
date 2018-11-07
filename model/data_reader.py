import json
import numpy as np
import torch as t
import torch.nn as nn
from torch.utils.data import Dataset
from collections import namedtuple
from nltk.tokenize import WordPunctTokenizer
import bisect
from model.util import get_device
from model.predictor import PredictorConfig, GRUConfig, DocQAPredictor
from model.corpus import CorpusStats
from model.trainer import Trainer
from model.tokenizer import NltkTokenizer
from model.text_processor import TextProcessor
from model.corpus import TrainDataset, EvalDataset
from model.wv import WordVectors as OldWordVectors

Data = namedtuple(
    "Data",
    [
        "paras",
        "qid_to_para",
        "qid_to_q",
        "tokenized_paras",
        "qid_to_tokenized_q",
        "encoded_paras",
        "qid_to_encoded_qtext",
        "qid_to_encoded_ans",
    ],
)

WordVectors = namedtuple("WordVectors", ["vectors", "word_to_idx", "idx_to_word"])

VectorEncodedData = namedtuple(
    "VectorEncodedData",
    ["qids", "paras", "qid_to_q", "qid_to_para", "qid_to_ans", "vectors"],
)

Mapping = namedtuple(
    "Mapping",
    [
        "all_tokens",
        "all_chars",
        "token_to_id",
        "id_to_token",
        "char_to_id",
        "id_to_char",
    ],
)

EncodedSample = namedtuple(
    "EncodedSample",
    [
        "question_words",
        "question_id",
        "context_words",
        "span_starts",
        "span_ends",
        "question_chars",
        "context_chars",
    ],
)

Token = namedtuple("Token", ["words", "spans"])


def tokenize(text):
    text = text.lower()
    spans = list(WordPunctTokenizer().span_tokenize(text))
    words = [text[span_start:span_end] for span_start, span_end in spans]
    return Token(words, spans)


def read_raw_dict(data_file):
    data = Data([], {}, {}, [], {}, [], {}, {})
    with open(data_file, "r") as f:
        data_dict = json.load(f)
    for doc in data_dict["data"]:
        json_paragraphs = doc["paragraphs"]
        for para in json_paragraphs:
            try:
                ctx = para["context"]
            except TypeError:
                continue
            data.paras.append(ctx)
            cur_para_idx = len(data.paras) - 1
            qas = para["qas"]
            for qa in qas:
                data.qid_to_para[qa["id"]] = cur_para_idx
                answer = (
                    qa["answers"][0]["answer_start"],
                    qa["answers"][0]["answer_start"] + len(qa["answers"][0]["text"]),
                )
                question = {"text": qa["question"], "answer": answer}
                data.qid_to_q[qa["id"]] = question
    return data


def tokenize_data(data, mapping, build_new_mapping):
    if build_new_mapping:
        mapping = Mapping(set(), set(), {}, {}, {}, {})
    for para in data.paras:
        ptkn = tokenize(para)
        if build_new_mapping:
            mapping.all_tokens.update(ptkn.words)
            mapping.all_chars.update("".join(ptkn.words))
        data.tokenized_paras.append(ptkn)

    for qid, qobj in data.qid_to_q.items():
        qtkn = tokenize(qobj["text"])
        if build_new_mapping:
            mapping.all_tokens.update(qtkn.words)
            mapping.all_chars.update("".join(qtkn.words))

        pid = data.qid_to_para[qid]
        ptkn = data.tokenized_paras[pid]
        ptkn_starts, ptkn_ends = zip(*ptkn.spans)
        ans_first_tkn = bisect.bisect_right(ptkn_starts, qobj["answer"][0]) - 1
        ans_last_tkn = bisect.bisect_left(ptkn_ends, qobj["answer"][1])
        data.qid_to_tokenized_q[qid] = {
            "tokens": qtkn.words,
            "answer": (ans_first_tkn, ans_last_tkn),
        }
    return data, mapping


def build_mapping(mapping):
    mapping.id_to_token[1] = "<UNK>"
    mapping.id_to_char[1] = "<UNK>"
    mapping.token_to_id.update(dict(map(reversed, enumerate(mapping.all_tokens, 2))))
    mapping.id_to_token.update(dict(enumerate(mapping.all_tokens, 2)))
    mapping.char_to_id.update(dict(map(reversed, enumerate(mapping.all_chars, 2))))
    mapping.id_to_char.update(dict(enumerate(mapping.all_chars, 2)))
    return mapping


def encode_data(data, mapping):
    data = data._replace(
        encoded_paras=[
            [mapping.token_to_id.get(word, 1) for word in para.words]
            for para in data.tokenized_paras
        ]
    )
    for qid, qobj in data.qid_to_tokenized_q.items():
        encoded_question = [mapping.token_to_id.get(word, 1) for word in qobj["tokens"]]
        para_len = len(data.encoded_paras[data.qid_to_para[qid]])
        ans = qobj["answer"]
        encoded_answer_start = [1 if i == ans[0] else 0 for i in range(para_len)]
        encoded_answer_end = [1 if i == ans[1] else 0 for i in range(para_len)]
        data.qid_to_encoded_qtext[qid] = encoded_question
        data.qid_to_encoded_ans[qid] = (encoded_answer_start, encoded_answer_end)
    return data


def parse_squad(data_file, build_new_mapping=False, mapping=None):
    if build_new_mapping:
        mapping = Mapping(set(), set(), {}, {}, {}, {})
    data = read_raw_dict(data_file)
    data, mapping = tokenize_data(data, mapping, build_new_mapping)
    if build_new_mapping:
        mapping = build_mapping(mapping)
    data = encode_data(data, mapping)
    return data, mapping


def read_vectors_file(vectors_file):
    vectors_list = []
    vocab = []
    with open(vectors_file, "r") as f:
        for line in f:
            word, vec_data = line.rstrip("\n").split(" ", 1)
            vector = np.array(
                [float(num) for num in vec_data.split(" ")], dtype=np.float32
            )
            vectors_list.append(vector)
            vocab.append(word)
    pad_vec = np.zeros_like(vectors_list[0])
    vectors_list.insert(0, pad_vec)
    vocab.insert(0, "<PAD>")
    unk_vec = np.random.randn((vectors_list[0].shape[0]))
    vectors_list.insert(1, unk_vec)
    vocab.insert(1, "<UNK>")
    return WordVectors(
        vectors=np.stack(vectors_list),
        idx_to_word=dict(enumerate(vocab)),
        word_to_idx=dict(map(reversed, enumerate(vocab))),
    )


def encode_dataset_with_vectors(data, mapping, starting_vectors, filter_vectors=False):
    if filter_vectors:
        dataset_mapping_id_to_vectors_idx = {
            idx: starting_vectors.word_to_idx.get(word, 1)
            for idx, word in mapping.id_to_token.items()
        }
        dataset_mapping_id_to_vectors_idx.update({0: 0, 1: 1})  # pad->pad, unk->unk
        used_vector_indices = list(sorted(dataset_mapping_id_to_vectors_idx.values()))

        compact_vectors = np.take(starting_vectors.vectors, used_vector_indices, axis=0)
        word_to_compact_vector_idx = {}
        compact_vector_idx_to_word = {}
        for new_idx, old_idx in enumerate(used_vector_indices):
            word_to_compact_vector_idx[starting_vectors.idx_to_word[old_idx]] = new_idx
            compact_vector_idx_to_word[new_idx] = starting_vectors.idx_to_word[old_idx]
        vectors = WordVectors(
            vectors=compact_vectors,
            word_to_idx=word_to_compact_vector_idx,
            idx_to_word=compact_vector_idx_to_word,
        )
    else:
        vectors = starting_vectors

    dataset_mapping_id_to_vectors_idx = {
        idx: vectors.word_to_idx.get(word, 1)
        for idx, word in mapping.id_to_token.items()
    }
    encoded_qid_to_q = {
        qid: [dataset_mapping_id_to_vectors_idx[tok_id] for tok_id in qobj]
        for qid, qobj in data.qid_to_encoded_qtext.items()
    }
    encoded_paras = [
        [dataset_mapping_id_to_vectors_idx[tok_id] for tok_id in para]
        for para in data.encoded_paras
    ]

    vector_encoded_data = VectorEncodedData(
        qids=list(data.qid_to_para.keys()),
        paras=encoded_paras,
        qid_to_q=encoded_qid_to_q,
        qid_to_para=data.qid_to_para,
        qid_to_ans=data.qid_to_encoded_ans,
        vectors=vectors,
    )
    return vector_encoded_data


class SQACorpus:
    def __init__(self, encoded_data):
        self.stats = CorpusStats(
            n_contexts=len(encoded_data.paras),
            n_questions=len(encoded_data.qids),
            n_answerable=len(encoded_data.qids),
            n_unanswerable=0,
            max_context_len=max(len(para) for para in encoded_data.paras),
            max_q_len=max(len(qtext) for qtext in encoded_data.qid_to_q.values()),
            max_word_len=0,
            single_answer=True,
            word_vocab_size=0,
            char_vocab_size=0,
        )
        self.idx_to_word = encoded_data.vectors.idx_to_word
        self.encoded_data = encoded_data
        self.samples = []
        for qid in encoded_data.qids:
            q_words = np.array(encoded_data.qid_to_q[qid])
            ctx_words = np.array(encoded_data.paras[encoded_data.qid_to_para[qid]])
            span_starts = np.array(encoded_data.qid_to_ans[qid][0])
            span_ends = np.array(encoded_data.qid_to_ans[qid][1])
            q_chars = np.array((len(encoded_data.qid_to_q[qid]), 1))
            ctx_chars = np.array((ctx_words.shape[0], 1))
            self.samples.append(
                EncodedSample(
                    q_words, qid, ctx_words, span_starts, span_ends, q_chars, ctx_chars
                )
            )


class SQADataset(Dataset):
    def __init__(self, encoded_data):
        self.corpus = SQACorpus(encoded_data)
        self.source_file = "data/original/dev.json"

    def __len__(self):
        return len(self.corpus.samples)

    def __getitem__(self, idx):
        return self.corpus.samples[idx]

    def get_answer_texts(self, answer_token_idxs):
        return {
            qid: self.get_single_answer_text(qid, span_start, span_end)
            for qid, (span_start, span_end) in answer_token_idxs.items()
        }

    def get_single_answer_text(self, qid, span_start, span_end):
        # TODO: Should this be +1 for end
        return " ".join(
            self.corpus.encoded_data.paras[self.corpus.encoded_data.qid_to_para[qid]][
                span_start:span_end
            ]
        )


class WordEmbeddor(nn.Module):
    def __init__(self, vectors):
        super().__init__()
        self.embedding_dim = vectors.shape[1]
        embedding_matrix = t.Tensor(vectors)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)

    def forward(self, words, chars):
        return self.embedding(words)


def initialize_model(
    word_vecs,
    use_cuda=True,
    rnn_hidden_size=128,
    rnn_num_layers=2,
    dropout=0.1,
    bidirectional=True,
    attention_linear_hidden_size=128,
    use_self_attention=True,
    batch_size=32,
):
    device = get_device(not use_cuda)
    predictor_config = PredictorConfig(
        gru=GRUConfig(rnn_hidden_size, rnn_num_layers, dropout, not bidirectional),
        dropout_prob=dropout,
        attention_linear_hidden_size=attention_linear_hidden_size,
        use_self_attention=use_self_attention,
        batch_size=batch_size,
    )
    embeddor = WordEmbeddor(word_vecs)
    predictor_model = DocQAPredictor(embeddor, predictor_config).to(device)
    return predictor_model


def get_old_style_datasets(
    train_file="data/original/train.json",
    dev_file="data/original/dev.json",
    vector_file="data/word-vectors/glove/glove.6B.100d.txt",
):
    tokenizer = NltkTokenizer()
    processor = TextProcessor({"lowercase": True})
    vectors = OldWordVectors.load_vectors(vector_file)

    train_dataset = TrainDataset.load_dataset(
        train_file, vectors, tokenizer, processor, False
    )
    dev_dataset = EvalDataset.load_dataset(
        dev_file,
        train_dataset.token_mapping,
        train_dataset.char_mapping,
        tokenizer,
        processor,
    )
    return train_dataset, dev_dataset


def get_simple_datasets(
    train_file="data/original/train.json",
    dev_file="data/original/dev.json",
    vector_file="data/word-vectors/glove/glove.6B.100d.txt",
):
    train_data, mapping = parse_squad(train_file, build_new_mapping=True)
    dev_data, mapping = parse_squad(dev_file, mapping=mapping)
    glove_vectors = read_vectors_file(vector_file)
    glove_encoded_train_data = encode_dataset_with_vectors(
        train_data, mapping, glove_vectors, filter_vectors=True
    )
    train_vecs = glove_encoded_train_data.vectors
    glove_encoded_dev_data = encode_dataset_with_vectors(
        dev_data, mapping, train_vecs, filter_vectors=False
    )
    train_dataset = SQADataset(glove_encoded_train_data)
    dev_dataset = SQADataset(glove_encoded_dev_data)
    return train_dataset, dev_dataset


def train_model_with_new_dataset(
    train_file="data/original/train.json",
    dev_file="data/original/dev.json",
    vector_file="data/word-vectors/glove/glove.6B.100d.txt",
):
    train_dataset, dev_dataset = get_simple_datasets(train_file, dev_file, vector_file)

    test_model = initialize_model(
        train_dataset.corpus.encoded_data.vectors.vectors,
        use_cuda=True,
        rnn_hidden_size=64,
        rnn_num_layers=1,
        dropout=0,
        bidirectional=False,
        use_self_attention=False,
        batch_size=6,
    )

    training_config = Trainer.TrainingConfig(
        learning_rate=2e-3,
        weight_decay=1e-5,
        num_epochs=20,
        batch_size=32,
        max_question_size=500,
        max_context_size=500,
        device=get_device(False),
        loader_num_workers=0,
        model_checkpoint_path="jupyter.pth",
    )

    Trainer.train_model(
        test_model, train_dataset, dev_dataset, training_config, debug=False
    )


def compare_datasets(
    train_file="data/original/train.json",
    dev_file="data/original/dev.json",
    vector_file="data/word-vectors/glove/glove.6B.100d.txt",
):
    simple_train_dataset, simple_dev_dataset = get_simple_datasets(
        train_file, dev_file, vector_file
    )
    old_train_dataset, old_dev_dataset = get_old_style_datasets(
        train_file, dev_file, vector_file
    )
    assert len(simple_train_dataset) == len(old_train_dataset)
    for i in range(len(simple_train_dataset)):
        simple_sample = simple_train_dataset[i]
        old_sample = old_train_dataset[i]
        assert (
            simple_sample.question_words == old_sample.question_words
        ), "{} vs {}".format(
            simple_sample, old_sample
        )
        assert simple_sample.question_id == old_sample.question_id, "{} vs {}".format(
            simple_sample, old_sample
        )
        assert (
            simple_sample.context_words == old_sample.context_words
        ), "{} vs {}".format(
            simple_sample, old_sample
        )
        assert simple_sample.span_starts == old_sample.span_starts, "{} vs {}".format(
            simple_sample, old_sample
        )
        assert simple_sample.span_end == old_sample.span_ends, "{} vs {}".format(
            simple_sample, old_sample
        )

    assert len(simple_dev_dataset) == len(old_dev_dataset)
    for i in range(len(simple_dev_dataset)):
        simple_sample = simple_dev_dataset[i]
        old_sample = old_dev_dataset[i]
        assert (
            simple_sample.question_words == old_sample.question_words
        ), "{} vs {}".format(
            simple_sample, old_sample
        )
        assert simple_sample.question_id == old_sample.question_id, "{} vs {}".format(
            simple_sample, old_sample
        )
        assert (
            simple_sample.context_words == old_sample.context_words
        ), "{} vs {}".format(
            simple_sample, old_sample
        )
        assert simple_sample.span_starts == old_sample.span_starts, "{} vs {}".format(
            simple_sample, old_sample
        )
        assert simple_sample.span_end == old_sample.span_ends, "{} vs {}".format(
            simple_sample, old_sample
        )
