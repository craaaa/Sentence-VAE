import torch
import numpy as np
from torch.autograd import Variable
from collections import defaultdict, Counter, OrderedDict


class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def idx2word(idx, i2w, pad_idx, sep=" ", use_bert=False, bert_tokenizer=None):

    sent_str = [str()]*len(idx)
    sent_list = [[]]*len(idx)

    for i, sent in enumerate(idx):
        if use_bert:
            sent_str[i] = bert_tokenizer.decode(sent)
        else:
            sent_list[i] = []
            for word_id in sent:
                if word_id == pad_idx:
                    break
                word = i2w[str(word_id.item())]
                sent_str[i] += word + sep
                sent_list[i].append(word)
            sent_str[i] = sent_str[i].strip()

    return sent_str, sent_list


def idx2defandword(def_and_word, i2w, i2a, pad_idx, use_bert=False, bert_tokenizer=None):
    def_idx, word_idx = def_and_word
    def_string, def_list = idx2word(def_idx, i2w=i2w, pad_idx=pad_idx, use_bert=use_bert, bert_tokenizer=bert_tokenizer)
    def_string = process_ngram_string(def_list, n=2)
    word_string, word_list = idx2word(word_idx, i2w=i2a, pad_idx=pad_idx, sep="", use_bert=use_bert, bert_tokenizer=bert_tokenizer)
    word_string = process_ngram_string(word_list, n=2)
    #[:-5] removes the <eos> token at the end of the prediction
    return [word[:-5] + ": " + defn[:-5] for defn, word in zip(def_string, word_string)]

def process_ngram_string(ngram_list, n=2):
    def get_ngrams(ngram_list):
        final_result = ngram_list[0]
        for prev, current in zip(ngram_list[:-1], ngram_list[1:]):
            if current[:-1] == prev[1:]: # if ngrams match
                final_result += current[-1]
            else:
                final_result += ("/" + current)
        return final_result
    return [get_ngrams(n) for n in ngram_list]

def interpolate(start, end, steps):

    interpolation = np.zeros((start.shape[0], steps + 2))

    for dim, (s,e) in enumerate(zip(start,end)):
        interpolation[dim] = np.linspace(s,e,steps+2)

    return interpolation.T

def expierment_name(args, ts):

    exp_name = str()
    exp_name += "BS=%i_"%args.batch_size
    exp_name += "LR={}_".format(args.learning_rate)
    exp_name += "EB=%i_"%args.embedding_size
    exp_name += "%s_"%args.rnn_type.upper()
    exp_name += "HS=%i_"%args.hidden_size
    exp_name += "L=%i_"%args.num_layers
    exp_name += "BI=%i_"%args.bidirectional
    exp_name += "LS=%i_"%args.latent_size
    exp_name += "WD={}_".format(args.word_dropout)
    exp_name += "ANN=%s_"%args.anneal_function.upper()
    exp_name += "K={}_".format(args.k)
    exp_name += "X0=%i_"%args.x0
    exp_name += "TS=%s"%ts

    return exp_name
