from solver import Model
from data_loader import get_loader
from configs import get_config
from utils import Vocab
import os
import pickle
import re
import os


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    os.environ['TORCH_HOME'] = './.torch'
    config = get_config(mode='test', batch_size=1)

    print('Loading Vocabulary...')
    vocab = Vocab(lang="zh")
    vocab.load(config.word2id_path, config.id2word_path)
    print(f'Vocabulary size: {vocab.vocab_size}')

    config.vocab_size = vocab.vocab_size
    data_loader = get_loader(
        sentences=load_pickle(config.sentences_path),
        images=load_pickle(config.images_path),
        conv_img_length=load_pickle(config.images_len_path),
        conversation_length=load_pickle(config.conversation_length_path),
        sentence_length=load_pickle(config.sentence_length_path),
        vocab=vocab,
        shuffle=False)

    model = Model(config, vocab=vocab, is_train=False)

    model.build()
    model.generate_for_evaluation()
