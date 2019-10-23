import os
import io
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer

from utils import OrderedCounter

class PTB(Dataset):

    def __init__(self, data_dir, split, create_data, **kwargs):

        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 50)
        self.min_occ = kwargs.get('min_occ', 3)

        self.raw_definition_path = os.path.join(data_dir, split+'.def.txt')
        self.raw_word_path = os.path.join(data_dir, split+'.word.txt')
        self.data_file = split+'.json'
        self.vocab_file = 'vocab.json'

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
        return len(self.w2i)

    @property
    def alphabet_size(self):
        return len(self.a2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

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
            with open(os.path.join(self.data_dir, self.vocab_file), 'r') as file:
                vocab = json.load(file)
            self.w2i, self.i2w = vocab['w2i'], vocab['i2w']
            self.a2i, self.i2a = vocab['a2i'], vocab['i2a']

    def _load_vocab(self):
        with open(os.path.join(self.data_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)

        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']
        self.a2i, self.i2a = vocab['a2i'], vocab['i2a']

    def _create_data(self):

        if self.split == 'train':
            self._create_vocab()
        else:
            self._load_vocab()

        tokenizer = TweetTokenizer(preserve_case=False)

        data = defaultdict(dict)
        with open(self.raw_definition_path, 'r') as file:

            for i, line in enumerate(file):

                words = tokenizer.tokenize(line)

                input = ['<sos>'] + words
                input = input[:self.max_sequence_length]

                target = words[:self.max_sequence_length-1]
                target = target + ['<eos>']

                assert len(input) == len(target), "%i, %i"%(len(input), len(target))
                length = len(input)

                input.extend(['<pad>'] * (self.max_sequence_length-length))
                target.extend(['<pad>'] * (self.max_sequence_length-length))

                input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]
                target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]

                id = len(data)
                data[id]['input'] = input
                data[id]['target'] = target
                data[id]['length'] = length

        with open(self.raw_word_path, 'r') as file:

            for i, line in enumerate(file):

                words = list(line.strip())

                target = words[:self.max_sequence_length-1]
                target = target + ['<eos>']

                length = len(target)

                target.extend(['<pad>'] * (self.max_sequence_length-length))

                target = [self.w2i.get(w, self.a2i['<unk>']) for w in target]

                data[i]['word'] = target
                data[i]['word_length'] = length


        with io.open(os.path.join(self.data_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def _create_vocab(self):

        assert self.split == 'train', "Vocablurary can only be created for training file."

        tokenizer = TweetTokenizer(preserve_case=False)

        w2c = OrderedCounter()
        a2c = OrderedCounter()
        w2i = dict()
        i2w = dict()
        a2i = dict()
        i2a = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)
            i2a[len(a2i)] = st
            a2i[st] = len(a2i)

        with open(self.raw_definition_path, 'r') as file:

            for i, line in enumerate(file):
                words = tokenizer.tokenize(line)
                w2c.update(words)

            for w, c in w2c.items():
                if c > self.min_occ and w not in special_tokens:
                    i2w[len(w2i)] = w
                    w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)

        with open(self.raw_word_path, 'r') as file:

            for i, line in enumerate(file):
                words = list(line.strip())
                w2c.update(words)

        for w, c in w2c.items():
            if c > self.min_occ and w not in special_tokens:
                i2a[len(a2i)] = w
                a2i[w] = len(a2i)

        assert len(a2i) == len(i2a)

        print("Vocabulary of %i keys created." %len(w2i))
        print("Alphabet of %i keys created." % len(a2i))

        vocab = dict(w2i=w2i, i2w=i2w, a2i=a2i, i2a=i2a)
        with io.open(os.path.join(self.data_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        self._load_vocab()
