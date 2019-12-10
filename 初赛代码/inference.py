from train import get_train_data
import torch
import pickle
from LipModel import LipModel

num_class = 313
device = 'cuda:0'
batch_size = 4

def predict(model_path, data_path, vocab_path, result_to_save):
    load_chach = True
    ##############################
    #         模型加载
    ##############################
    model = LipModel(1, num_class)
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
    model.eval()
    model.to(device)
    print('加载模型')

    id2label = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for word in f:
            id2label.append(word.split(',')[0])

    ##############################
    #         数据加载
    ##############################
    if load_chach:
        with open('cache/test_cache.dat', 'rb') as f:
            test_data = pickle.load(f)
            test_ids = pickle.load(f)
        print('加载缓存数据')
    else:
        with open(data_path, 'rb') as f:
            test_data = pickle.load(f)
            test_ids = pickle.load(f)
        print('数据加载完成, data num = {}, label num = {}'.format(len(test_data), len(test_ids)))

        test_data, test_ids = get_train_data(test_data, test_ids, batch_size, test_data=True)
        with open('cache/test_catch.dat', 'wb') as f:
            pickle.dump(test_data, f)
            pickle.dump(test_ids, f)
        print('缓存数据')
    print('pad填充完成, test batch num = {}'.format(len(test_data)))

    ##############################
    #            预测
    ##############################
    print('预测中...')
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
    print('预测完成！predict logit shape={}'.format(len(pre_result)))

    with open(result_to_save, 'wb') as f:
        pickle.dump(pre_result, f)
    print('保存为：' + result_to_save + '\n')
    return pre_result

if __name__ == '__main__':
    temp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for k in temp:
        model_path = 'model/fold_{}_model.pt'.format(k)
        data_path = 'data/test_data.dat'
        vocab_path = 'data/vocab.txt'
        result_to_save = 'model/fold_{}_result.pkl'.format(k)
        predict(model_path, data_path, vocab_path, result_to_save)
