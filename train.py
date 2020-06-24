from solver import *
from data_loader import get_loader
from configs import get_config
from utils import Vocab
import os
import pickle
import torch.distributed as dist
import warnings
import random
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    config = get_config(mode='train')
    val_config = get_config(mode='valid', batch_size=32, workers=1)
    print(config)
    with open(os.path.join(config.save_path, 'config.txt'), 'w') as f:
        print(config, file=f)

    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if config.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if config.dist_url == "env://" and args.world_size == -1:
        config.world_size = int(os.environ["WORLD_SIZE"])

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if config.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.world_size = ngpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config, val_config))
    else:
        # Simply call main_worker function
        main_worker(config.gpu, ngpus_per_node, config, val_config)


def main_worker(gpu, ngpus_per_node, config, val_config):
    config.gpu = gpu

    if config.gpu is not None:
        print("Use GPU: {} for training".format(config.gpu))

    if config.distributed:
        if config.dist_url == "env://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.rank = config.rank * ngpus_per_node + gpu  # pid
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                world_size=config.world_size, rank=config.rank)

    print('Loading Vocabulary...')
    vocab = Vocab()
    vocab.load(config.word2id_path, config.id2word_path)
    print(f'Vocabulary size: {vocab.vocab_size}')

    config.vocab_size = vocab.vocab_size

    model = Model(config, vocab=vocab, is_train=True)
    model.build(ngpus_per_node=ngpus_per_node )

    train_sampler, train_data_loader = get_loader(
        sentences=load_pickle(config.sentences_path),
        images=load_pickle(config.images_path),
        conv_img_length=load_pickle(config.images_len_path),
        conversation_length=load_pickle(config.conversation_length_path),
        sentence_length=load_pickle(config.sentence_length_path),
        vocab=vocab,
        config=config,
        data_type='train'
    )

    eval_data_loader = get_loader(
        sentences=load_pickle(val_config.sentences_path),
        images=load_pickle(val_config.images_path),
        conv_img_length=load_pickle(val_config.images_len_path),
        conversation_length=load_pickle(val_config.conversation_length_path),
        sentence_length=load_pickle(val_config.sentence_length_path),
        vocab=vocab,
        config=val_config,
        data_type='eval'
    )

    model.train(train_sampler, train_data_loader, eval_data_loader)


if __name__ == '__main__':
    main()
