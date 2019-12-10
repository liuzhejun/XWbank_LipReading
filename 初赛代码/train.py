# -*-coding:utf-8-*-
import numpy as np
import random
import torch
import os
import pickle
# from LipModel import LipModel
from LipModel3 import LipModel
from multiprocessing.pool import Pool
from multiprocessing import Process
from tqdm import tqdm
import argparse

def padding_batch(array_batch):
    '''
    将一个batch的样本填充至同样帧数
    :param array_batch: 一个batch大小的样本
    :return: 一个batch的训练数据 tensor shape: (batch_size, 1, time_steps, h, w)
    '''
    data = []
    time_steps = [a.shape[0] for a in array_batch]
    max_timestpe = max(time_steps)
    for i, array in enumerate(array_batch):
        if array.shape[0] != max_timestpe:
            t, h, w, c = array.shape
            pad_arr = np.zeros((max_timestpe-t, h, w, c), dtype=np.float32)
            array_batch[i] = np.vstack((array, pad_arr))
            logging('padding data size:{}'.format(pad_arr.shape))
        data.append(array_batch[i])
    data = np.asarray(data, dtype=np.float32)
    data = data.transpose((0, 4, 1, 2, 3))
    return data

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def logging(s, log_path='log.txt', print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+', encoding='utf-8') as f_log:
            f_log.write(s + '\n')


def get_train_data(array_list, label_list, batch_size, test_data=False):
    '''
    将由单个样本组成的训练数据处理成批次数据
    :param array_list: 训练数据列表
    :param label_list: 标签列表
    :param batch_size: batch size
    :param test_data: 是否是测试数据
    :return:
    '''
    label_data = []
    num_data = len(array_list)
    num_batch = num_data // batch_size if num_data % batch_size == 0 else num_data // batch_size + 1

    batch_range = list(range(num_batch))
    random.shuffle(batch_range)
    bar = tqdm(batch_range)

    iter_list = []
    for i in bar:
        start = i * batch_size
        end = (i + 1) * batch_size if (i + 1) * batch_size < num_data else num_data
        iter_list.append(array_list[start:end])

        if test_data:
            label_data.append(label_list[start:end])
        else:
            label_data.append(torch.tensor(label_list[start:end]))
    bar.close()

    pool = Pool()
    train_data = pool.map(padding_batch, iter_list)
    pool.close()
    pool.join()
    train_data = [torch.tensor(data_item) for data_item in train_data]

    return train_data, label_data

def split_train_eval(array_list, label_list, num_eval):
    '''
    分割训练集和验证集，若是经过分割了的k折交叉检验数据，则无需再继续分割
    :param array_list:
    :param label_list:
    :param num_eval:
    :return:
    '''
    train_data = []
    train_label = []
    eval_data = []
    eval_label = []
    eval_idx = random.sample(range(len(array_list)), num_eval)
    for i in range(len(array_list)):
        if i not in eval_idx:
            train_data.append(array_list[i])
            train_label.append(label_list[i])
        else:
            eval_data.append(array_list[i])
            eval_label.append(label_list[i])
    return train_data, train_label, eval_data, eval_label


def save_data(fname, *datas):
    with open(fname, 'wb') as f:
        for data in datas:
            pickle.dump(data, f)
    print('数据保存完成:{}'.format(fname))


def new_process_to_save_data(fname, *datas):
    p = Process(target=save_data, args=(fname, *datas))
    p.start()


def eval(model, eval_data, eval_label, device):
    '''
    验证
    :param model:
    :param eval_data:
    :param eval_label:
    :param device:
    :return:
    '''
    model.eval()
    acc = 0
    count = 0
    with torch.no_grad():
        for step in range(len(eval_data)):
            batch_inputs = eval_data[step].to(device)
            batch_labels = eval_label[step].to(device)

            logist = model(batch_inputs)[0]
            count += logist.size(0)
            acc += torch.sum(torch.eq(torch.argmax(logist, dim=-1), batch_labels)).item()
    model.train()
    return acc/count

def predict(model, k, batch_size, model_path, data_path, vocab_path, result_to_save, device):
    '''
    预测
    :param model: 模型
    :param k: 第几折
    :param batch_size: batch size
    :param model_path: 最好模型保存路径
    :param data_path: 测试集路径
    :param vocab_path: 词表路径
    :param result_to_save: 预测结果要保存的位置
    :param device: 设备
    :return:
    '''
    logging('*'*48 + '预测' + '*'*48)
    # 之后第一次预测需要处理测试集数据，之后可以直接加载缓存
    load_cache = False if k == 0 else True
    ##############################
    #         模型加载
    ##############################
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
    model.eval()
    model.to(device)
    logging('加载模型')

    id2label = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for word in f:
            id2label.append(word.split(',')[0])

    ##############################
    #         数据加载
    ##############################
    if load_cache:
        with open('cache/test_cache.dat', 'rb') as f:
            test_data = pickle.load(f)
            test_ids = pickle.load(f)
        logging('加载缓存数据')
    else:
        with open(data_path, 'rb') as f:
            test_data = pickle.load(f)
            test_ids = pickle.load(f)
        logging('数据加载完成, data num = {}, label num = {}'.format(len(test_data), len(test_ids)))
        test_data, test_ids = get_train_data(test_data, test_ids, batch_size, test_data=True)

        if not os.path.exists('cache/'):
            os.mkdir('cache/')

        with open('cache/test_cache.dat', 'wb') as f:
            pickle.dump(test_data, f)
            pickle.dump(test_ids, f)
        logging('缓存数据')
    logging('pad填充完成, test batch num = {}'.format(len(test_data)))

    ##############################
    #            预测
    ##############################
    # logging('预测中...')
    # pre_result = []
    # with torch.no_grad():
    #     for step in range(len(test_data)):
    #         batch_inputs = test_data[step].to(device)
    #         logist = model(batch_inputs)[0]
    #
    #         pred = torch.argmax(logist, dim=-1).tolist()
    #         assert len(pred) == len(test_ids[step])
    #         for i, ids in enumerate(test_ids[step]):
    #             pre_result.append(ids + ',' + id2label[pred[i]])
    # with open(result_to_save, 'w', encoding='utf-8') as f:
    #     for line in pre_result:
    #         f.write(line + '\n')
    # logging('预测结果已保存至:', result_to_save)

    pre_result = {}
    with torch.no_grad():
        for step in range(len(test_data)):
            batch_inputs = test_data[step].to(device)
            logist = model(batch_inputs)[0]

            for i, ids in enumerate(test_ids[step]):
                if ids in pre_result:
                    pre_result[ids] += logist[i]
                else:
                    pre_result[ids] = logist[i]
    logging('预测完成！predict logit shape={}'.format(len(pre_result)))
    with open(result_to_save, 'wb') as f:
        pickle.dump(pre_result, f)
    logging('预测结果保存至:' + result_to_save)
    logging('*' * 100 + '\n')
    return pre_result



def train(args, k):
    num_class = 313
    save_model = True

    test_data_path = args.test_data_path
    vocab_path = args.vocab_path
    model_save_path = args.model_save_path
    batch_size = args.batch_size
    epochs = args.epochs
    device = 'cuda:0'
    lr = args.lr
    log_step = args.log_step
    grad_clip = args.grad_clip
    num_eval = args.num_eval
    eval_batch = args.eval_batch
    load_cache = args.load_cache

    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)


    ##############################
    #         模型加载
    ##############################
    model = LipModel(3, num_class)
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    logging('加载模型')
    num_param = get_parameter_number(model)
    logging('total parameter: {}, trainable parameter: {}'.format(num_param['Total'], num_param['Trainable']))

    ##############################
    #         数据加载
    ##############################
    data_path = os.path.join(args.data_path, 'data_fold_{}.dat'.format(k))
    with open(data_path, 'rb') as f:
        train_data = pickle.load(f)
        train_label = pickle.load(f)
        eval_data = pickle.load(f)
        eval_label = pickle.load(f)
    assert len(train_data) == len(train_label) and len(eval_data) == len(eval_label)
    logging('数据加载完成, data num = {}, eval num = {}'.format(len(train_data), len(eval_data)))

    ##############################
    #         数据处理
    ##############################
    if load_cache:
        with open('cache/cache.dat', 'rb') as f:
            train_data = pickle.load(f)
            train_label = pickle.load(f)
            eval_data = pickle.load(f)
            eval_label = pickle.load(f)
        logging('加载缓存数据')
    else:
        if not os.path.exists('cache/'):
            os.mkdir('cache/')
        train_data, train_label = get_train_data(train_data, train_label, batch_size)
        eval_data, eval_label = get_train_data(eval_data, eval_label, eval_batch)
        print('缓存数据...')
        new_process_to_save_data('cache/cache.dat', train_data, train_label, eval_data, eval_label)
        # with open('cache/cache.dat', 'wb') as f:
        #     pickle.dump(train_data, f)
        #     pickle.dump(train_label, f)
        #     pickle.dump(eval_data, f)
        #     pickle.dump(eval_label, f)
        # logging('缓存数据')
    logging('pad填充完成, train batch num = {}, eval batch num = {}'.format(len(train_data), len(eval_data)))

    ##############################
    #            训练
    ##############################
    best_acc = -1
    pred_label = []
    true_label = []
    for epoch in range(1, epochs+1):
        avg_loss = 0
        data_indexs = list(range(len(train_data)))
        random.shuffle(data_indexs)
        for step, data_idx in enumerate(data_indexs):
            batch_inputs = train_data[data_idx].to(device)
            batch_labels = train_label[data_idx].to(device)

            logist, loss = model(batch_inputs, batch_labels)
            logist = torch.argmax(logist, dim=-1)
            loss = loss.mean()

            pred_label.append(logist)
            true_label.append(batch_labels)
            avg_loss += loss.item()
            if step % log_step == 0:
                pred_acc = torch.mean(torch.eq(torch.cat(pred_label), torch.cat(true_label)).float()).item()
                logging('epoch={}, step={}, timestep={}, loss={:.3f}, pred acc={:.3f}'.format(
                    epoch, step, batch_inputs.size(2), avg_loss/(step+1), pred_acc))
                pred_label = []
                true_label = []

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            step += 1
        acc = eval(model, eval_data, eval_label, device=device)
        logging('='*100)
        logging('epoch = {}, Avg train loss = {}, Acc = {}'.format(epoch, avg_loss/len(train_data), acc))

        if save_model and acc >= best_acc:
            model_to_save = model.module if hasattr(model, 'module') else model
            model_save_path = os.path.join(args.model_save_path, 'fold_{}_model.pt'.format(k))
            with open(model_save_path, 'wb') as f:
                torch.save(model_to_save.state_dict(), f)
            logging('保存模型:' + model_save_path)
            best_acc = acc
        logging('=' * 100)
    logging('训练完成: best acc ={}'.format(best_acc) + '\n')


    result_to_save = os.path.join(args.model_save_path, 'fold_{}_result.pkl'.format(k))
    pred_logist = predict(model, k, eval_batch, model_save_path, test_data_path,
                          vocab_path=vocab_path, result_to_save=result_to_save, device=device)
    return pred_logist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='data/', type=str,
                        help='train data path')
    parser.add_argument("--test_data_path", default='data/test_data.dat', type=str,
                        help='test data path')
    parser.add_argument("--vocab_path", default='data/vocab.txt', type=str,
                        help='vocab path')
    parser.add_argument("--model_save_path", default='model/', type=str,
                        help='the path model to save')
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--device", default='cuda:0', type=str)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--grad_clip", default=0.5, type=float)

    parser.add_argument("--log_step", default=100, type=int, help='print information interval')
    parser.add_argument("--num_eval", default=5000, type=int, help='number of verification set')
    parser.add_argument("--eval_batch", default=4, type=int, help='batch size of verify')
    parser.add_argument('--load_cache', action='store_true')
    parser.add_argument("--k", default=10, type=int, help='跑k次')
    args = parser.parse_args()

    logging('*第1折*')
    pred_logist = train(args, 0)
    for k in range(1, args.k):
        logging('*第{}折*'.format(k+1))
        new_pred_logist = train(args, k)
        for key in new_pred_logist.keys():
            pred_logist[key] += new_pred_logist[key]

    id2label = []
    with open(args.vocab_path, 'r', encoding='utf-8') as f:
        for word in f:
            id2label.append(word.split(',')[0])

    pre_result = []
    for key in pred_logist.keys():
        pred_word = id2label[torch.argmax(pred_logist[key]).item()]
        pre_result.append(key + ',' + pred_word)
    with open('submit.csv', 'w', encoding='utf-8') as f:
        for line in pre_result:
            f.write(line + '\n')
    logging('预测结果已保存至:' + 'submit.csv')
