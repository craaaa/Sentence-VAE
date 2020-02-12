import os
import json
import time
import torch
import argparse
import numpy as np
import utils
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

from ptb import PTB
from utils import to_var, idx2word, expierment_name
from model import SentenceVAE

def main(args):

    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.localtime())

    splits = ['train', 'valid'] + (['test'] if args.test else [])

    datasets = OrderedDict()
    for split in splits:
        datasets[split] = PTB(
            data_dir=args.data_dir,
            split=split,
            create_data=args.create_data,
            max_sequence_length=args.max_sequence_length,
            min_occ=args.min_occ,
            use_bert=False
        )

    model = SentenceVAE(
        alphabet_size=datasets['train'].alphabet_size,
        vocab_size=datasets['train'].vocab_size,
        sos_idx=datasets['train'].sos_idx,
        eos_idx=datasets['train'].eos_idx,
        pad_idx=datasets['train'].pad_idx,
        unk_idx=datasets['train'].unk_idx,
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

    if torch.cuda.is_available():
        model = model.cuda()

    print(model)

    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.logdir, expierment_name(args,ts)))
        writer.add_text("model", str(model))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)

    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path)
    print ("Saving model to directory: " + save_model_path)

    def kl_anneal_function(anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            return float(1/(1+np.exp(-k*(step-x0))))
        elif anneal_function == 'linear':
            return min(1, step/x0)

    def word_weight_function(step, k, x0):
        return float(1/(1+np.exp(-k*(step-x0))))

    NLL = torch.nn.NLLLoss(reduction='sum', ignore_index=datasets['train'].pad_idx)

    def loss_fn(def_logp, word_logp, def_target, def_length, word_target, word_length, mean, logv):

        # cut-off unnecessary padding from target definition, and flatten
        def_target = def_target[:,:torch.max(def_length).item()].contiguous().view(-1)
        def_logp = def_logp.view(-1, def_logp.size(2))

        # Negative Log Likelihood
        def_NLL_loss = NLL(def_logp, def_target)

        # cut off padding for words
        word_target = word_target[:, :torch.max(word_length).item()].contiguous().view(-1)
        word_logp = word_logp.view(-1, word_logp.size(2))

        # Word NLL
        word_NLL_loss = NLL(word_logp, word_target)

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())

        return def_NLL_loss, word_NLL_loss, KL_loss

    def get_weights(anneal_function, step, k, x0):
        # for logistic function, k = growth rate
        KL_weight = kl_anneal_function(anneal_function, step, k, x0)
        word_weight = word_weight_function(step, k, x0)

        return  {'def': 1,
                 'word': word_weight,
                 'kl': KL_weight
                }

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0
    for epoch in range(args.epochs):

        for split in splits:
            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                shuffle=split=='train',
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )

            tracker = defaultdict(tensor)

            # Enable/Disable Dropout
            if split == 'train':
                model.train()
            else:
                model.eval()

            for iteration, batch in enumerate(data_loader):

                batch_size = batch['input'].size(0)

                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                # Forward pass
                [def_logp, word_logp], mean, logv, z = model(batch['input'], batch['length'], batch['word_length'])

                # loss calculation
                def_NLL_loss, word_NLL_loss, KL_loss = loss_fn(def_logp, word_logp, batch['target'], batch['length'], batch['word'], batch['word_length'], mean, logv)
                weights = get_weights(args.anneal_function, step, args.k, args.x0)

                loss = (weights['def'] * def_NLL_loss +
                        weights['word'] * word_NLL_loss +
                        weights['kl'] * KL_loss)/batch_size

                mean_logv = torch.mean(logv)

                # backward + optimization
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1


                # bookkeepeing
                tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.detach().unsqueeze(0)))

                if args.tensorboard_logging:
                    writer.add_scalar("%s/ELBO"%split.upper(), loss.item(), epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/Def NLL Loss"%split.upper(), def_NLL_loss.item()/batch_size, epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/Word NLL Loss"%split.upper(), word_NLL_loss.item()/batch_size, epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL Loss"%split.upper(), KL_loss.item()/batch_size, epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL Weight"%split.upper(), weights['kl'], epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/Word Weight"%split.upper(), weights['word'], epoch*len(data_loader) + iteration)

                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    print("%s Batch %04d/%i, Loss %9.4f, Def NLL-Loss %9.4f, Word NLL-Loss %9.4f  Word-Weight %6.3f, KL-Loss %9.4f, KL-Weight %6.3f KL-VAL %9.4f"
                        %(split.upper(), iteration, len(data_loader)-1, loss.item(), def_NLL_loss.item()/batch_size, word_NLL_loss.item()/batch_size, weights['word'], KL_loss.item()/batch_size, weights['kl'], mean_logv))

                if split == 'valid':
                    if 'target_sents' not in tracker:
                        tracker['target_sents'] = list()
                    tracker['target_sents'] += idx2word(batch['target'], i2w=datasets['train'].get_i2w(), pad_idx=datasets['train'].pad_idx,
                        use_bert=args.use_bert,
                        bert_tokenizer=datasets['valid'].tokenizer)
                    tracker['z'] = torch.cat((tracker['z'], z.data), dim=0)

            print("%s Epoch %02d/%i, Mean ELBO %9.4f"%(split.upper(), epoch, args.epochs, torch.mean(tracker['ELBO'])))

            if args.tensorboard_logging:
                writer.add_scalar("%s-Epoch/ELBO"%split.upper(), torch.mean(tracker['ELBO']), epoch)

            # save a dump of all sentences and the encoded latent space
            if split == 'valid':
                dump = {'target_sents':tracker['target_sents'], 'z':tracker['z'].tolist()}
                if not os.path.exists(os.path.join('dumps', ts)):
                    os.makedirs('dumps/'+ts)
                with open(os.path.join('dumps/'+ts+'/valid_E%i.json'%epoch), 'w') as dump_file:
                    json.dump(dump,dump_file)

            # save checkpoint
            if split == 'train':
                checkpoint_path = os.path.join(save_model_path, "E%i.pytorch"%(epoch))
                torch.save(model.state_dict(), checkpoint_path)
                print("Model saved at %s"%checkpoint_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=60)
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('--test', action='store_true')

    parser.add_argument('-ep', '--epochs', type=int, default=4)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    parser.add_argument('-eb', '--embedding_size', type=int, default=768)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    parser.add_argument('-bert', '--use_bert', action='store_true', default=False)

    parser.add_argument('-af', '--anneal_function', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)

    parser.add_argument('-v','--print_every', type=int, default=50)
    parser.add_argument('-tb','--tensorboard_logging', action='store_true')
    parser.add_argument('-log','--logdir', type=str, default='logs')
    parser.add_argument('-bin','--save_model_path', type=str, default='bin')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    args.anneal_function = args.anneal_function.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert args.anneal_function in ['logistic', 'linear']
    assert 0 <= args.word_dropout <= 1

    main(args)
