import json
import numpy as np
from torch.utils.data import Dataset
from collections import namedtuple
from nltk.tokenize import WordPunctTokenizer
import bisect
from typing import Dict, List, Tuple, Any, Optional

from model.corpus import CorpusStats

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

Token = namedtuple("Tokenized", ["words", "spans"])
tokenizer = WordPunctTokenizer()


def tokenize(text: str) -> Token:
    text = text.lower()
    spans = list(tokenizer.span_tokenize(text))
    words = [text[span_start:span_end] for span_start, span_end in spans]
    return Token(words, spans)


def read_raw_dict(data_file: str) -> Data:
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


def tokenize_data(
    data: Data, mapping: Optional[Mapping], build_new_mapping: bool
) -> Tuple[Data, Mapping]:
    if mapping is None:
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


def build_mapping(mapping: Mapping) -> Mapping:
    mapping.id_to_token[1] = "<UNK>"
    mapping.id_to_char[1] = "<UNK>"
    mapping.token_to_id.update(dict(map(reversed, enumerate(mapping.all_tokens, 2))))
    mapping.id_to_token.update(dict(enumerate(mapping.all_tokens, 2)))
    mapping.char_to_id.update(dict(map(reversed, enumerate(mapping.all_chars, 2))))
    mapping.id_to_char.update(dict(enumerate(mapping.all_chars, 2)))
    return mapping


def encode_data(data: Data, mapping: Mapping) -> Data:
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


def parse_squad(
    data_file: str, build_new_mapping: bool = False, mapping: Optional[Mapping] = None
) -> Tuple[Data, Mapping]:
    data = read_raw_dict(data_file)
    data, mapping = tokenize_data(data, mapping, build_new_mapping)
    if build_new_mapping:
        mapping = build_mapping(mapping)
    data = encode_data(data, mapping)
    return data, mapping


def read_vectors_file(vectors_file: str) -> WordVectors:
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


def encode_dataset_with_vectors(
    data: Data,
    mapping: Mapping,
    starting_vectors: WordVectors,
    filter_vectors: bool = False,
) -> VectorEncodedData:
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

    stats: CorpusStats
    idx_to_word: Dict[int, str]
    encode_data: VectorEncodedData
    samples: List[EncodedSample]

    def __init__(self, encoded_data: VectorEncodedData) -> None:
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

    corpus: SQACorpus
    source_file: str

    def __init__(self, encoded_data: VectorEncodedData) -> None:
        self.corpus = SQACorpus(encoded_data)
        self.source_file = "data/original/dev.json"

    def __len__(self) -> int:
        return len(self.corpus.samples)

    def __getitem__(self, idx: int) -> EncodedSample:
        return self.corpus.samples[idx]

    def get_answer_texts(
        self, answer_token_idxs: Dict[str, Tuple[int, int]]
    ) -> Dict[str, str]:
        return {
            qid: self.get_single_answer_text(qid, span_start, span_end)
            for qid, (span_start, span_end) in answer_token_idxs.items()
        }

    def get_single_answer_text(self, qid: str, span_start: int, span_end: int) -> str:
        # TODO: Should this be +1 for end
        return " ".join(
            self.corpus.encoded_data.paras[self.corpus.encoded_data.qid_to_para[qid]][
                span_start:span_end
            ]
        )


def read_batchable_dataset(
    train_dataset_file: str, dev_dataset_file: str, word_vector_file: str
) -> Tuple[SQADataset, SQADataset]:
    train_data, mapping = parse_squad(train_dataset_file, build_new_mapping=True)
    dev_data, mapping = parse_squad(dev_dataset_file, mapping=mapping)
    vectors = read_vectors_file(word_vector_file)
    vector_encoded_train_data = encode_dataset_with_vectors(
        train_data, mapping, vectors, filter_vectors=True
    )
    vector_encoded_dev_data = encode_dataset_with_vectors(
        dev_data, mapping, vector_encoded_train_data.vectors, filter_vectors=False
    )
    train_dataset = SQADataset(vector_encoded_train_data)
    dev_dataset = SQADataset(vector_encoded_dev_data)
    return train_dataset, dev_dataset
