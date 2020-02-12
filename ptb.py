import os
import io
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer
from nltk.util import ngrams
from transformers import BertTokenizer

from utils import OrderedCounter

class PTB(Dataset):

    def __init__(self, data_dir, split, create_data, use_bert=False, **kwargs):

        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 50)
        self.min_occ = kwargs.get('min_occ', 3)

        self.raw_definition_path = os.path.join(data_dir, split+'.def.txt')
        self.raw_word_path = os.path.join(data_dir, split+'.word.txt')
        self.data_file = split+'.json'
        self.vocab_file = 'vocab.json'

        self.use_bert = use_bert
        if self.use_bert:
        # if True:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = TweetTokenizer(preserve_case=False)


        if create_data:
            print("Creating new %s data."%split.upper())
            self._create_data()

        elif not os.path.exists(os.path.join(self.data_dir, self.data_file)):
            print("%s preprocessed file not found at %s. Creating new."%(split.upper(), os.path.join(self.data_dir, self.data_file)))
            self._create_data()

        else:
            self._load_data()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)

        return {
            'input': np.asarray(self.data[idx]['input']),
            'target': np.asarray(self.data[idx]['target']),
            'word': np.asarray(self.data[idx]['word']),
            'length': self.data[idx]['length'],
            'word_length': self.data[idx]['word_length']
        }

    @property
    def vocab_size(self):
        if self.use_bert:
            return self.tokenizer.vocab_size
        else:
            return len(self.w2i)

    @property
    def alphabet_size(self):
        if self.use_bert:
            return self.tokenizer.vocab_size
        else:
            return len(self.a2i)

    @property
    def pad_idx(self):
        return self.w2i[self.pad]

    @property
    def sos_idx(self):
        return self.w2i[self.sos]

    @property
    def eos_idx(self):
        return self.w2i[self.eos]

    @property
    def unk_idx(self):
        return self.w2i[self.unk]

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w

    def get_a2i(self):
        return self.a2i

    def get_i2a(self):
        return self.i2a


    def _load_data(self, vocab=True):

        with open(os.path.join(self.data_dir, self.data_file), 'r') as file:
            self.data = json.load(file)
        if vocab:
            self._load_vocab()

    def _load_vocab(self):
        with open(os.path.join(self.data_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)

        self.pad = '[PAD]'
        self.unk = '[UNK]'
        self.sos = '[CLS]'
        self.eos = '[SEP]'

        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']
        self.a2i, self.i2a = vocab['a2i'], vocab['i2a']

    def _create_data(self):

        if self.split == 'train':
            self._create_vocab()
        else:
            self._load_vocab()

        data = defaultdict(dict)
        with open(self.raw_definition_path, 'r') as file:

            for i, line in enumerate(file):

                # words = self.tokenizer.tokenize(line)
                words = [''.join(a) for a in ngrams(line.strip(), 2)]

                input = [self.sos] + words
                if self.use_bert: # add [SEP] to end of input
                    input = input[:self.max_sequence_length-1] + [self.eos]
                    target = list(input)
                else:
                    input = input[:self.max_sequence_length]
                    target = words[:self.max_sequence_length-1] + [self.eos]

                assert len(input) == len(target), "%i, %i"%(len(input), len(target))
                length = len(input)

                input.extend([self.pad] * (self.max_sequence_length-length))
                target.extend([self.pad] * (self.max_sequence_length-length))

                if self.use_bert:
                    input = self.tokenizer.convert_tokens_to_ids(input)
                    target = self.tokenizer.convert_tokens_to_ids(target)
                else:
                    input = [self.w2i.get(w, self.w2i[self.unk]) for w in input]
                    target = [self.w2i.get(w, self.w2i[self.unk]) for w in target]

                id = len(data)
                data[id]['input'] = input
                data[id]['target'] = target
                data[id]['length'] = length

        with open(self.raw_word_path, 'r') as file:

            for i, line in enumerate(file):

                words = list(line.strip())

                target = words[:self.max_sequence_length-1]
                target = target + [self.eos]

                length = len(target)

                target.extend([self.pad] * (self.max_sequence_length-length))

                if self.use_bert:
                    target = self.tokenizer.convert_tokens_to_ids(target)
                else:
                    target = [self.a2i.get(w, self.a2i[self.unk]) for w in target]

                data[i]['word'] = target
                data[i]['word_length'] = length


        with io.open(os.path.join(self.data_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def _create_vocab(self):

        assert self.split == 'train', "Vocablurary can only be created for training file."

        w2c = OrderedCounter()
        a2c = OrderedCounter()
        w2i = dict()
        i2w = dict()
        a2i = dict()
        i2a = dict()

        self.pad = '[PAD]'
        self.unk = '[UNK]'
        self.sos = '[CLS]'
        self.eos = '[SEP]'

        special_tokens = [self.pad, self.unk, self.sos, self.eos]
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)
            i2a[len(a2i)] = st
            a2i[st] = len(a2i)

        with open(self.raw_definition_path, 'r') as file:

            for i, line in enumerate(file):
                # words = self.tokenizer.tokenize(line)
                words = [''.join(a) for a in ngrams(line.strip(), 2)]
                w2c.update(words)

            for w, c in w2c.items():
                if c > self.min_occ and w not in special_tokens:
                    i2w[len(w2i)] = w
                    w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)

        with open(self.raw_word_path, 'r') as file:

            for i, line in enumerate(file):
                words = list(line.strip())
                a2c.update(words)

        for w, c in a2c.items():
            if c > self.min_occ and w not in special_tokens:
                i2a[len(a2i)] = w
                a2i[w] = len(a2i)

        assert len(a2i) == len(i2a)

        if self.use_bert:
            vocab_size = self.tokenizer.vocab_size
            alph_size = self.tokenizer.vocab_size
        else:
            vocab_size = len(w2i)
            alph_size = len(a2i)

        print("Vocabulary of %i keys created." % vocab_size)
        print(list(w2i.keys())[:100])
        print("Alphabet of %i keys created." % alph_size)
        print(list(a2i.keys()))


        vocab = dict(w2i=w2i, i2w=i2w, a2i=a2i, i2a=i2a, vocab_size=vocab_size, alph_size=alph_size)
        with io.open(os.path.join(self.data_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        self._load_vocab()
