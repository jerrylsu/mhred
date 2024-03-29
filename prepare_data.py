import os
from multiprocessing import Pool
import argparse
import pickle
import random
from pathlib import Path
from tqdm import tqdm
from utils import Vocab, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN

# set default path for data and test data
project_dir = Path(__file__).resolve().parent
datasets_dir = project_dir.joinpath('./data/')
images_train_dir = project_dir.joinpath('./data/images_train/')
images_valid_dir = project_dir.joinpath('./data/images_dev/')
images_test_dir = project_dir.joinpath('./online_test_data/images_test/')


def load_conversations(fileName, spliter="</s>"):
    conversations = []
    images = []
    with open(fileName, 'r', encoding='UTF-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fs = line.split("\t")
            if len(fs) != 3:
                print("error line", line)

            context, response = fs[0].strip(), fs[1].strip()
            utterances = context.split(spliter)
            conversation = []
            for utterance in utterances:
                conversation.append(utterance.split())
            conversation.append(response.split())
            conversations.append(conversation)
            images.append(fs[2].strip())
    return conversations, images


def pad_sentences(conversations, max_sentence_length=40, max_conversation_length=10):
    def pad_tokens(tokens, max_sentence_length=max_sentence_length):
        n_valid_tokens = len(tokens)
        if n_valid_tokens > max_sentence_length - 1:
            tokens = tokens[:max_sentence_length - 1]
        n_pad = max_sentence_length - n_valid_tokens - 1
        tokens = tokens + [EOS_TOKEN] + [PAD_TOKEN] * n_pad
        return tokens

    def pad_conversation(conversation):
        conversation = [pad_tokens(sentence) for sentence in conversation]
        return conversation

    all_padded_sentences = []
    all_sentence_length = []

    for conversation in conversations:
        if len(conversation) > max_conversation_length:
            conversation.reverse()
            conversation = conversation[:max_conversation_length]
            conversation.reverse() # the last n utterances
        sentence_length = [min(len(sentence) + 1, max_sentence_length) # +1 for EOS token
                           for sentence in conversation]
        all_sentence_length.append(sentence_length)

        sentences = pad_conversation(conversation)
        all_padded_sentences.append(sentences)

    sentences = all_padded_sentences
    sentence_length = all_sentence_length
    return sentences, sentence_length


def images_str_2_list(dir, images, max_conv_length):
    all_img_list = list()
    all_img_len_list = list()

    for image in images:
        img_list = image.strip().split(' ')

        name_list = ['NULL']*max_conv_length
        mark_list = [0]*max_conv_length

        img_list.reverse()
        img_list = img_list[:max_conv_length]
        img_list.reverse()

        for idx, item in enumerate(img_list):
            if item == 'NULL':
                None
            else:
                name_list[idx] = str(dir)+"/"+item
                mark_list[idx] = 1

        all_img_list.append(name_list)
        all_img_len_list.append(mark_list)

    return all_img_list, all_img_len_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Maximum valid length of sentence
    # => SOS/EOS will surround sentence (EOS for source / SOS for target)
    # => maximum length of tensor = max_sentence_length + 1
    parser.add_argument('-s', '--max_sentence_length', type=int, default=50)
    parser.add_argument('-c', '--max_conversation_length', type=int, default=10)

    # Vocabulary
    parser.add_argument('--max_vocab_size', type=int, default=50000)
    parser.add_argument('--min_vocab_frequency', type=int, default=3)

    args = parser.parse_args()

    max_sent_len = args.max_sentence_length
    max_conv_len = args.max_conversation_length
    max_vocab_size = args.max_vocab_size
    min_freq = args.min_vocab_frequency

    print("Loading conversations...")
    train, train_img = load_conversations(datasets_dir.joinpath("train.txt"))
    valid, valid_img = load_conversations(datasets_dir.joinpath("dev.txt"))

    print("#train=%d, #val=%d" % (len(train), len(valid)))

    def to_pickle(obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    vocab = Vocab(lang="zh")
    for split_type, conversations, images in [('train', train, train_img), ('valid', valid, valid_img)]:
        print(f'Processing {split_type} dataset...')
        split_data_dir = datasets_dir.joinpath(split_type)
        split_data_dir.mkdir(exist_ok=True)
        conversation_length = [min(len(conv), max_conv_len)
                               for conv in conversations]

        sentences, sentence_length = pad_sentences(
            conversations,
            max_sentence_length=max_sent_len,
            max_conversation_length=max_conv_len)

        if split_type == 'train':
            images_dir = images_train_dir
        elif split_type == 'valid':
            images_dir = images_valid_dir
        elif split_type == 'test':
            images_dir = images_test_dir

        images, images_length = images_str_2_list(images_dir, images, max_conv_length=max_conv_len)
        print('Saving preprocessed data at', split_data_dir)
        to_pickle(conversation_length, split_data_dir.joinpath('conversation_length.pkl'))
        to_pickle(sentences, split_data_dir.joinpath('sentences.pkl'))
        to_pickle(sentence_length, split_data_dir.joinpath('sentence_length.pkl'))
        to_pickle(images, split_data_dir.joinpath('images.pkl'))
        to_pickle(images_length, split_data_dir.joinpath('images_length.pkl'))

        if split_type != 'test':
            print('Save Vocabulary...')
            vocab.add_dataframe(conversations)
            vocab.update(max_size=max_vocab_size, min_freq=min_freq)

            print('Vocabulary size: ', len(vocab))
            vocab.pickle(datasets_dir.joinpath('word2id.pkl'), datasets_dir.joinpath('id2word.pkl'))

    print('Done!')
