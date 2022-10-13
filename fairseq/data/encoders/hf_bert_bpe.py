# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional

from fairseq.data.encoders import register_bpe
from fairseq.dataclass import FairseqDataclass


@dataclass
class BertBPEConfig(FairseqDataclass):
    bpe_cased: bool = field(default=False, metadata={"help": "set for cased BPE"})
    bpe_vocab_file: Optional[str] = field(
        default=None, metadata={"help": "bpe vocab file"}
    )

@register_bpe("bert", dataclass=BertBPEConfig)
class BertBPE(object):
    def __init__(self, cfg):
        try:
            from transformers import BertTokenizer
        except ImportError:
            raise ImportError(
                "Please install transformers with: pip install transformers"
            )
        import pdb;pdb.set_trace()
        if cfg.bpe_vocab_file:
            self.bert_tokenizer = BertTokenizer(
                cfg.bpe_vocab_file, do_lower_case=not cfg.bpe_cased
            )
        else:
            vocab_file_name = (
                "bert-base-cased" if cfg.bpe_cased else "bert-base-uncased"
            )
            self.bert_tokenizer = BertTokenizer.from_pretrained(vocab_file_name)

    def encode(self, x: str) -> str:
        return " ".join(self.bert_tokenizer.tokenize(x))

    def decode(self, x: str) -> str:
        return self.bert_tokenizer.clean_up_tokenization(
            self.bert_tokenizer.convert_tokens_to_string(x.split(" "))
        )

    def is_beginning_of_word(self, x: str) -> bool:
        return not x.startswith("##")



def get_sentence(tokens):
    sentence = []
    length = len(tokens)
    i=0
    while(i < length):
        index = [i]
        index = find_pound_key(tokens, i, index)
        i = index[-1]+1
        word = [tokens[j].strip("##") for j in index]
        word = "".join(word)
        sentence.append(word)
    return sentence

def find_pound_key(tokens, i, index):
    if i == len(tokens)-1:
        return index
    if "##" not in tokens[i+1]:
        return index
    if "##" in tokens[i+1]:
        index.append(i+1)
        return find_pound_key(tokens, i+1, index)


@dataclass
class BiBertBPEConfig(BertBPEConfig):
    pretrained_bpe: Optional[str] = field(
        default=None, metadata={"help": "pre-trained tgt bpe model"})
    pretrained_bpe_src: Optional[str] = field(
        default=None, metadata={"help": "pre-trained src bpe model"}
    )


@register_bpe("bibert", dataclass=BiBertBPEConfig)
class BiBertBPE(object):
    def __init__(self, cfg, if_src=False):
        try:
            from transformers import AutoTokenizer, BertTokenizer
        except ImportError:
            raise ImportError(
                "Please install transformers with: pip install transformers"
            )
        self.pretrained_bpe = cfg.pretrained_bpe
        self.pretrained_bpe_src = cfg.pretrained_bpe_src
        if not if_src:
            self.bert_tokenizer = BertTokenizer.from_pretrained(cfg.pretrained_bpe)
        else:
            self.bert_tokenizer = BertTokenizer.from_pretrained(cfg.pretrained_bpe_src)



    def encode(self, x: str) -> str:
        return " ".join(self.bert_tokenizer.tokenize(x))

    def decode(self, x: str, if_src=False) -> str:
        return " ".join(get_sentence(x.split(" ")))

    def is_beginning_of_word(self, x: str) -> bool:
        return not x.startswith("##")
