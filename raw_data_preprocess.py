# -*- coding: utf-8 -*-

import argparse
import re
import jieba

pattern_pun = '！，；：？、。"!,;:?."\''
pattern_jpg = re.compile(r'[A-Za-z0-9]+\.jpg')


def clean_text(text):
    text = re.sub(r'[{}]+'.format(r'\d+\*\*'), '<num>', text)
    text = re.sub(r'[{}]+'.format(r'\d+'), '<num>', text)
    # text = clean_punctuation(text)
    return text


def clean_punctuation(text):
    text = re.sub(r'[{}]+'.format(pattern_pun), '', text)
    return text.strip().lower()


def tokenize_spt(text):
    sp_token = ['<img>', '<url>', '<sos>', '<eos>', '<num>']
    resp_list = list()
    tmp_list = jieba.cut(text, cut_all=False)
    seg_list = list(tmp_list)
    i = 0
    while i < len(seg_list):
        if ''.join(seg_list[i:i + 3]) in sp_token:
            resp_list.append(''.join(seg_list[i:i + 3]))
            i = i + 3
        else:
            resp_list.append(''.join(seg_list[i]))
            i = i + 1
    return resp_list


class DataIterm(object):
    def __init__(self, sid, ques, ans, ctx):
        self.sid = sid
        self.ques = ques
        self.ans = ans
        self.ctx = ctx


def do_preprocess(dir, sess_turn):
    """
    :param dir:  官方数据存放路径
    :param sess_turn: context中保存的历史上下文的对话轮数
    :return: train_items, dev_items
             用于训练的train和dev数据，其中每条数据记录由以下几部分原始信息组成
             sid, 对话原始的session信息，后续按照需要可以根据该信息查询对话相关的知识库，本实例中未使用
             question, 该条训练数据所对应的用户的问题
             answer, 该条训练数据对应的客服的回答
             context, 该对话发生的上下文信息，该信息最大信息长度不超过sess_turn所定义的轮数
    """
    sess_len = sess_turn * 2
    train_items, dev_items = list(), list()
    for file, item_list in [('data_train.txt', train_items), ('data_dev.txt', dev_items)]:
        with open(dir + file, 'r', encoding='UTF-8') as f:
            lines = f.readlines()

        data_list, sess_pid = list(), dict()
        for line in lines:
            words = line.strip().split('\t')
            sid, shop, pid, text, waiter = words[0], words[1], words[2], words[3], words[4]
            if pid:
                sess_pid[sid] = pid
            text = 'A: ' + text if waiter == '1' else 'Q: ' + text
            data_list.append((sid, text))

        data_len = len(data_list)
        i = 0
        tmp_data_list = list()

        # 将原始数据按照session和问题、回答类型，
        # 用'|||'连接不同回车发送的内容
        while i < data_len:
            head_i, text_i, sid_i = data_list[i][1][0], data_list[i][1], data_list[i][0]
            j = i + 1
            if j >= data_len:
                tmp_data_list.append((sid_i, text_i))
                break
            head_j, text_j, sid_j = data_list[j][1][0], data_list[j][1], data_list[j][0]
            add = 0
            while head_i == head_j and sid_i == sid_j:
                text_i = text_i + '|||' + text_j[3:]  # delete 'A: ' or 'Q: '
                add = add + 1
                j = j + 1
                if j >= data_len:
                    break
                head_j, text_j, sid_j = data_list[j][1][0], data_list[j][1], data_list[j][0]

            i = i + add + 1
            tmp_data_list.append((sid_i, text_i))

        # 遍历全部（session, Q: xxx） (session, A: xxx),
        # 构建训练输入文件，Q，A，Context，
        # 其中'@@@'间隔Context里面不同的Q或者A
        for idx, item in enumerate(tmp_data_list):
            sid, text = item[0], item[1]
            if text.startswith('A'):
                continue
            question = text.replace('Q: ', '').strip()
            if question == '':
                continue
            if idx + 1 >= len(tmp_data_list):
                continue
            n_item = tmp_data_list[idx + 1]
            n_sid = n_item[0]
            if sid != n_sid:
                continue
            n_text = n_item[1]
            answer = n_text.replace('A: ', '').strip()
            if answer == '':
                continue
            if idx > sess_len:
                cand_data_list = tmp_data_list[idx - sess_len:idx]
            else:
                cand_data_list = tmp_data_list[:idx]

            contxt_list = list()
            for cand_item in cand_data_list:
                cand_sid = cand_item[0]
                cand_text = cand_item[1]
                if cand_sid != sid:
                    continue
                contxt_list.append(cand_text)
            context = '@@@'.join(contxt_list)
            item_list.append(DataIterm(sid, question, answer, context))
    return train_items, dev_items


def gen_train_dev_set(dir, train_items, dev_items):
    """train_items
    ans = '正常的呢'
    ctx = 'Q: d0d070d46eeb27d794a661f182afed15.jpg|||售后咨询组@@@A: 您好，欢迎光临***官方旗舰店，麻烦您将需要咨询的问题简单说明下哦，以便能够快速的为您查看处理哈'
    ques = '63da14cfaaf83d2ec43ac2e28a35b216.jpg|||48bb0208de1279b8eb5465143fdebc31.jpg|||鞋上这么多胶水|||正常吗'
    sid = '79ca5475f50c1457031b5b312ef49d58'
    """
    f_train_out = open(dir + 'train.txt', 'w', encoding='UTF-8')
    f_dev_out = open(dir + 'dev.txt', 'w', encoding='UTF-8')

    for type in ['train', 'dev']:
        if type == 'train':
            items, f_out = train_items, f_train_out
        elif type == 'dev':
            items, f_out = dev_items, f_dev_out

        for item in items:
            src_str, trg_str, img_list = '', '', list()
            ques, ans, ctx = item.ques.strip(), item.ans.strip(), item.ctx

            # 1. handle context
            ctx_list = ctx.split('@@@')
            for sent_i in ctx_list:
                if sent_i == '':
                    continue
                sent_i_type = sent_i[0]
                sent_i = sent_i[3:].strip()
                sent_i_list = sent_i.split('|||')
                for sent_j in sent_i_list:
                    if sent_j.endswith('.jpg'):
                        img_list.append(sent_j)
                        sent_j = '<img>'
                    else:
                        img_list.append('NULL')
                        sent_j = clean_text(sent_j)
                    sent_seg = ' '.join(tokenize_spt(sent_j.strip()))
                    if sent_seg:
                        src_str = src_str + sent_seg + '</s>'
                    else:
                        img_list.pop(-1)

            # 2. handle question: 上下文context + 加上当前轮question = src_str
            ques_list = ques.split('|||')
            for sent in ques_list:
                if sent.endswith('.jpg'):
                    img_list.append(sent)
                    sent = '<img>'
                else:
                    img_list.append('NULL')
                    sent = clean_text(sent)

                sent = sent.strip()
                if sent:
                    sent_seg = ' '.join(tokenize_spt(sent.strip()))
                    src_str = src_str + sent_seg + '</s>'
                else:
                    img_list.pop(-1)

            # 3. handle answer: 只有当前轮当前轮answer = trg_str
            ans_list = ans.split('|||')
            for sent in ans_list:
                if sent.endswith('jpg'):
                    sent = '<img>'
                else:
                    sent = clean_text(sent)
                trg_str = trg_str + ' ' + ' '.join(tokenize_spt(sent.strip()))

            src_str = src_str[:-4]  # delete the </s> of tail
            trg_str = trg_str.strip()
            img_str = ' '.join(img_list)

            src_list = src_str.split('</s>')
            assert len(src_list) == len(img_list)

            if '<img>' not in trg_str:  # predict the sigle multimodal.
                f_out.write(src_str + '\t' + trg_str + '\t' + img_str + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="tool to process raw data")
    parser.add_argument('-d', '--directory', default='data/')
    parser.add_argument('-s', '--sess_turns', default=2)  # context中保存的历史上下文的对话轮数
    args = parser.parse_args()

    train_items, dev_items = do_preprocess(args.directory, args.sess_turns)
    gen_train_dev_set(args.directory, train_items, dev_items)
