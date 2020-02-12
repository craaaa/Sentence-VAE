import os
import json
import torch
import argparse

from collections import defaultdict
from model import SentenceVAE
from multiprocessing import cpu_count
from ptb import PTB
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from utils import to_var, idx2defandword, interpolate


def main(args):

    with open(args.data_dir+'/vocab.json', 'r') as file:
        vocab = json.load(file)

    w2i, i2w = vocab['w2i'], vocab['i2w']
    a2i, i2a = vocab['a2i'], vocab['i2a']
    vocab_size = vocab['vocab_size']
    alphabet_size = vocab['alph_size']


    model = SentenceVAE(
        vocab_size=vocab_size,
        alphabet_size=alphabet_size,
        sos_idx=w2i['[CLS]'],
        eos_idx=w2i['[SEP]'],
        pad_idx=w2i['[PAD]'],
        unk_idx=w2i['[UNK]'],
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
        )

    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s"%(args.load_checkpoint))

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    if args.test_file:
        test_dataset = PTB(
            data_dir=args.data_dir,
            split="test",
            create_data=True,
            max_sequence_length=args.max_sequence_length,
            min_occ=args.min_occ,
            use_bert=args.use_bert
        )

        data_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )
        tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        tracker = defaultdict(tensor)


        for iteration, batch in enumerate(data_loader):

            batch_size = batch['input'].size(0)

            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = to_var(v)

            # Forward pass
            [def_logp, word_logp], mean, logv, z = model(batch['input'], batch['length'], batch['word_length'])

            samples, z = model.inference(z=z)

            word_and_def = idx2defandword(samples, i2w=i2w, i2a=i2a, pad_idx=w2i['[PAD]'], use_bert=args.use_bert, bert_tokenizer=bert_tokenizer)
            word_and_def = [line + "\n" for line in word_and_def]

            with open(args.load_checkpoint + ".test", "a") as f:
                f.writelines(word_and_def)

    else:
        samples, z = model.inference(n=args.num_samples)
        print('----------SAMPLES----------')
        print(*idx2defandword(samples, i2w=i2w, i2a=i2a, pad_idx=w2i['[PAD]'], use_bert=args.use_bert, bert_tokenizer=bert_tokenizer), sep='\n', )

        z1 = torch.randn([args.latent_size]).numpy()
        z2 = torch.randn([args.latent_size]).numpy()
        z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float())
        samples, _ = model.inference(z=z)
        print('-------INTERPOLATION-------')
        print(*idx2defandword(samples, i2w=i2w, i2a=i2a, pad_idx=w2i['[PAD]'], use_bert=args.use_bert, bert_tokenizer=bert_tokenizer), sep='\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)
    parser.add_argument('-n', '--num_samples', type=int, default=10)

    parser.add_argument('-dd', '--data_dir', type=str, default='data')
    parser.add_argument('-ms', '--max_sequence_length', type=int, default=50)
    parser.add_argument('-eb', '--embedding_size', type=int, default=768)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-bert', '--use_bert', action='store_true')

    parser.add_argument('-tt', '--test_file', action='store_true')
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)


    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert 0 <= args.word_dropout <= 1

    main(args)
