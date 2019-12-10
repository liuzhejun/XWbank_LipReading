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
    for i in range(16):
        result_path = os.path.join(root_dir, 'fold_{}_result.pkl'.format(i))
        with open(result_path, 'rb') as f:
            kfold_result = pickle.load(f)

            for key in kfold_result.keys():
                if i <= 0:
                    result[key] = kfold_result[key].cpu()
                else:
                    result[key] += kfold_result[key].cpu()
        print(result_path, '完成')

    pred_result = []
    for key in result:
        pred_word = '{:0>3d}'.format(torch.argmax(result[key]).item())
        pred_result.append(key + ',' + pred_word)

    with open('submit2.csv', 'w', encoding='utf-8') as f:
        for line in pred_result:
            f.write(line + '\n')
    print('预测结果已保存至:', 'submit2.csv')

