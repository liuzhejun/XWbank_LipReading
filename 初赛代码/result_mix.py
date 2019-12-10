import os
import torch
import pickle
import numpy as np

if __name__ == '__main__':
    root_dir = 'model/'
    # result_path = os.path.join(root_dir, 'fold_0_result.pkl')
    # with open(result_path, 'rb') as f:
    #     fold_i_result = pickle.load(f)
    result = {}
    for i in range(10):
        result_path = os.path.join(root_dir, 'fold_{}_result.pkl'.format(i))
        with open(result_path, 'rb') as f:
            kfold_result = pickle.load(f)

            for key in kfold_result.keys():
                if i <= 0:
                    result[key] = kfold_result[key]
                else:
                    result[key] += kfold_result[key]

    id2label = []
    with open('data/vocab.txt', 'r', encoding='utf-8') as f:
        for word in f:
            id2label.append(word.split(',')[0])

    pred_result = []
    for key in result:
        pred_word = id2label[torch.argmax(result[key]).item()]
        pred_result.append(key + ',' + pred_word)

    with open('submit2.csv', 'w', encoding='utf-8') as f:
        for line in pred_result:
            f.write(line + '\n')
    print('预测结果已保存至:', 'submit2.csv')

