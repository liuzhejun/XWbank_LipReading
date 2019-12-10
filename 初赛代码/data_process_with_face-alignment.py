import os
import random
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Process
import face_alignment
from skimage import io, transform
mouth_bias_x = 20
mouth_bias_y = 20
norm_width = 180
norm_height = 120
span_width = int(norm_width/2)
span_height = int(norm_height/2)
# 每个样本在图像增强后产生几个样本（1表示未数据增强）
enhance_num_per_sample = 1


def check_available_fname(fname):
    try:
        if fname.split('.')[0].isdigit():
            return True
        else:
            return False
    except:
        return False

def normlize_image(img):
    img -= np.mean(img)
    img /= np.std(img)
    return img.astype(np.float32)


def computer_mouth_area(landmarks):
    mouth_width = 0
    mouth_heigth = 0
    mouth_info = []
    for lm in landmarks:
        mouth_lm = lm[48:]
        max_x, max_y, min_x, min_y = np.max(mouth_lm[:, 0]), np.max(mouth_lm[:, 1]), \
                                     np.min(mouth_lm[:, 0]), np.min(mouth_lm[:, 1])
        width_cut, height_cut = max_x - min_x, max_y - min_y
        center_width, center_height = int((max_x + min_x) / 2), int((max_y + min_y) / 2)
        # 每张图片的嘴唇中心位置
        mouth_info.append((center_height, center_width))
        # 所有图片中最大嘴唇区域面积
        mouth_width = width_cut if width_cut > mouth_width else mouth_width
        mouth_heigth = height_cut if height_cut > mouth_heigth else mouth_heigth

    return int(mouth_heigth), int(mouth_width), mouth_info



def clip_mouth(imgs, lms):
    assert len(imgs) == len(lms)
    cliped_imgs = []
    # 计算这组图片的嘴唇区域信息
    mouth_heigth, mouth_width, mouth_info = computer_mouth_area(lms)
    span_height = mouth_heigth // 2 + mouth_bias_y
    span_width = mouth_width //2 + mouth_bias_x

    for i, img in enumerate(imgs):
        center_height, center_width = mouth_info[i]
        start_height = center_height - span_height if center_height - span_height > 0 else 0
        start_width = center_width - span_width if center_width - span_width > 0 else 0
        end_heigth = center_height + span_height
        end_width = center_width + span_width

        mouthimg = img[start_height:end_heigth, start_width:end_width]
        mouthimg = transform.resize(mouthimg, (norm_height, norm_width))
        # io.imsave('temp/{}.png'.format(i), mouthimg)
        cliped_imgs.append(normlize_image(mouthimg))
    return cliped_imgs

def clip_mouth_with_enhance(imgs, lms):
    assert len(imgs) == len(lms)
    cliped_imgs = []
    translations = []
    reverse_imgs = []

    # 随机偏移量
    random_y = random.choice([-5, -4, -3, -2, 2, 3, 4, 5])
    random_x = random.choice([-5, -4, -3, -2, 2, 3, 4, 5])

    for i, img in enumerate(imgs):
        mouth_lm = lms[i][48:]
        max_x, max_y, min_x, min_y = np.max(mouth_lm[:, 0]), np.max(mouth_lm[:, 1]), \
                                     np.min(mouth_lm[:, 0]), np.min(mouth_lm[:, 1])
        center_width, center_height = int((max_x + min_x) / 2), int((max_y + min_y) / 2)
        width_cut, height_cut = max_x - min_x, max_y - min_y

        if width_cut > norm_width or height_cut > norm_height:
            # 原图
            cuty = int(min_y) - mouth_bias_y if int(min_y) - mouth_bias_y > 0 else 0
            cutx = int(min_x) - mouth_bias_x if int(min_x) - mouth_bias_x > 0 else 0
            mouthimg = img[cuty:int(max_y) + mouth_bias_y, cutx:int(max_x) + mouth_bias_x]
            mouthimg = transform.resize(mouthimg, (norm_height, norm_width))
            # 平移图
            trans_y = cuty + random_y if cuty + random_y > 0 else 0
            trans_x = cutx + random_x if cutx + random_x > 0 else 0
            trans_img = img[trans_y:int(max_y) + mouth_bias_y + random_y,
                            trans_x:int(max_x) + mouth_bias_x + random_x]
            trans_img = transform.resize(trans_img, (norm_height, norm_width))
            # 翻转图
            reverse_img = np.flip(mouthimg, axis=-1)
        else:
            # 原图
            cutheight = center_height - span_height if center_height - span_height > 0 else 0
            cutwidth = center_width - span_width if center_width - span_width > 0 else 0
            mouthimg = img[cutheight:center_height + span_height, cutwidth:center_width + span_width]
            mouthimg = transform.resize(mouthimg, (norm_height, norm_width))
            # 平移图
            trans_y = cutheight + random_y if cutheight + random_y > 0 else 0
            trans_x = cutwidth + random_x if cutwidth + random_x > 0 else 0
            trans_img = img[trans_y:center_height + span_height + random_y,
                            trans_x:center_width + span_width + random_x]
            trans_img = transform.resize(trans_img, (norm_height, norm_width))
            # 翻转图
            reverse_img = np.flip(mouthimg, axis=-1)
        assert mouthimg.shape == (norm_height, norm_width)
        assert trans_img.shape == (norm_height, norm_width)
        assert reverse_img.shape == (norm_height, norm_width)

        cliped_imgs.append(normlize_image(mouthimg))
        translations.append(normlize_image(trans_img))
        reverse_imgs.append(normlize_image(reverse_img))
    return cliped_imgs, translations, reverse_imgs

def split_k_flod(data_len, k):
    '''
    产生k折中每一折的训练集和验证集所对应的下标
    如输入：data__len=6, k=3
    输出：train_ids: [[2,3,4,5], [0,1,4,5], [0,1,2,3]]
          eval_ids: [[0,1], [2,3], [4,5]]
    :param data_len: 训练数据数量
    :param k: k折交叉检验
    :return:
    '''
    data_ids = set(range(data_len))
    train_ids = []
    eval_ids = []
    eval_data_num = data_len // k
    for i in range(k):
        if i == k-1:
            evals = data_ids
        else:
            evals = random.sample(data_ids, eval_data_num)
            data_ids -= set(evals)
        eval_ids.append(evals)
    for i in range(k):
        train_ids.append([])
        for j in range(k):
            if i != j:
                train_ids[i].extend(eval_ids[j])
    for i in range(k):
        print('Flod {}, len train_dirs = {}, len eval_dirs = {}'.format(k, len(train_ids), len(eval_ids)))
    return train_ids, eval_ids


def read_asample_imgs(dir_path):
    imgs = []
    img_paths = os.listdir(dir_path)
    img_paths = [int(i.split('.')[0]) for i in img_paths if i.split('.')[0].isdigit()]
    if len(img_paths) < 1:
        print('文件夹为空：{}'.format(dir_path))
        return None
    img_paths = ['{}.png'.format(i) for i in sorted(img_paths)]

    for img_path in img_paths:
        imgs.append(io.imread(os.path.join(dir_path, img_path), as_gray=False))
    return imgs


def read_data(landmarks, id2word=None, word2label=None, data_type='train'):
    data = []
    labels = []
    print('读取并裁剪{}数据:'.format(data_type))
    bar = tqdm(range(len(landmarks)))

    for sample_path in landmarks.keys():
        bar.update(1)
        imgs = read_asample_imgs(sample_path)
        if landmarks[sample_path] is None or imgs is None:
            print('样本被舍弃:{}'.format(sample_path))
            continue
        mouth_imgs = clip_mouth(imgs=imgs, lms=landmarks[sample_path])
        mouths_arr = np.array(mouth_imgs, dtype=np.float32)

        data.append(mouths_arr)
        sample_id = sample_path.split('/')[-1]
        if data_type == 'train':
            labels.append(word2label[id2word[sample_id]])
        else:
            labels.append(sample_id)
    bar.close()

    return data, labels


def shuffle_data(data, labels):
    data_ids = list(range(len(data)))
    random.shuffle(data_ids)
    sorted_idx = np.argsort([data[i].shape[0] for i in data_ids])
    sorted_data = []
    sorted_label = []
    for i in sorted_idx:
        sorted_data.append(data[data_ids[i]])
        sorted_label.append(labels[data_ids[i]])
    return sorted_data, sorted_label

def save_flod_data(enhance_data, labels, train_ids, eval_ids, save_path):
    '''
    保存某一折的训练集和验证集，训练集需数据增强，验证集不要数据增强
    :param enhance_data: 数据增强后的整个训练集
    :param labels: 数据增强后的整个训练集标签
    :param train_ids: 该折所对应的训练集数据在enhance_data中的下标集合
    :param eval_ids: 该折所对应的验证集数据在enhance_data中的下标集合
    :param save_path: 保存路径
    :return: None
    '''
    enhance_train_ids = []
    enhance_eval_ids = []
    # 将训练集和验证集未数据增强扩展之前的id转化为 数据增强扩展之后的id， 对应于enhance_data和labels的id
    for idx in train_ids:
        for i in range(enhance_num_per_sample):
            enhance_train_ids.append(idx*enhance_num_per_sample + i)
    for idx in eval_ids:
        enhance_eval_ids.append(idx*enhance_num_per_sample)

    print('shuffle train data...')
    random.shuffle(enhance_train_ids)
    sorted_idx = np.argsort([enhance_data[i].shape[0] for i in enhance_train_ids])
    sorted_train_data = []
    sorted_train_label = []
    for i in sorted_idx:
        sorted_train_data.append(enhance_data[enhance_train_ids[i]])
        sorted_train_label.append(labels[enhance_train_ids[i]])

    print('shuffle eval data...')
    random.shuffle(enhance_eval_ids)
    sorted_idx = np.argsort([enhance_data[i].shape[0] for i in enhance_eval_ids])
    sorted_eval_data = []
    sorted_eval_label = []
    for i in sorted_idx:
        sorted_eval_data.append(enhance_data[enhance_eval_ids[i]])
        sorted_eval_label.append(labels[enhance_eval_ids[i]])

    print('保存中...')
    with open(save_path, 'wb') as f:
        pickle.dump(sorted_train_data, f)
        pickle.dump(sorted_train_label, f)
        pickle.dump(sorted_eval_data, f)
        pickle.dump(sorted_eval_label, f)

def get_vocab(label_file):
    '''
    建立词表
    :param label_file: lip_train.txt文件的位置
    :return: 样本id与词语的对应id2word, 词语与下标的对应word2label
    '''
    id2word = {}
    word2label = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            ids, word = line.strip().split('\t')
            id2word[ids] = word
            if word not in word2label:
                word2label[word] = len(word2label)
    return id2word, word2label

def save_vocab(word2label, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for word in word2label.keys():
            f.write('{},{}\n'.format(word, word2label[word]))



def clip_mouth_(img_path):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True, device='cuda:0')
    img = io.imread(img_path)
    lms = fa.get_landmarks(img)
    if not lms:
        print('could not detect face!')
        return None
    mouth_lm = lms[0][48:]
    max_x, max_y, min_x, min_y = np.max(mouth_lm[:, 0]), np.max(mouth_lm[:, 1]), np.min(mouth_lm[:, 0]), np.min(mouth_lm[:, 1])
    center_width, center_height = int((max_x + min_x) / 2), int((max_y + min_y) / 2)
    width_cut, height_cut = max_x - min_x, max_y - min_y

    if width_cut > norm_width or height_cut > norm_height:
        print('有效区域大于指定区域大小')
        # 标准化区域小于有效区域，以有效区域切割，并且进行尺度放缩归一化
        cuty = int(min_y) - mouth_bias_y if int(min_y) - mouth_bias_y > 0 else 0
        cutx = int(min_x) - mouth_bias_x if int(min_x) - mouth_bias_x > 0 else 0
        mouthimg = img[cuty:int(max_y) + mouth_bias_y, cutx:int(max_x) + mouth_bias_x, :]
        mouthimg = transform.resize(mouthimg, (norm_height, norm_width))
    else:
        cutheight = center_height - span_height if center_height - span_height > 0 else 0
        cutwidth = center_width - span_width if center_width - span_width > 0 else 0
        mouthimg = img[cutheight:center_height + span_height, cutwidth:center_width + span_width, :]
    return mouthimg

if __name__ == '__main__':
    # mouth = clip_mouth_('xin_data/train_dataset/lip_train/6d8f8ab244b07fe8f8ca24aee5ac16f5/5.png')
    # io.imsave('1.png', mouth)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default='hack_data/train/hack_lip_train/', type=str,
                        help='lip_train folder path')
    parser.add_argument("--test_path", default='hack_data/test/hack_lip_test/', type=str,
                        help='lip_test folder path')
    parser.add_argument("--label_path", default='hack_data/train/hack_lip_train.txt', type=str,
                        help='lip_train.txt file path')
    parser.add_argument("--train_landmarks_path", default='data/train_landmarks.dat', type=str,
                        help='landmarks data file path of train data')
    parser.add_argument("--test_landmarks_path", default='data/test_landmarks.dat', type=str,
                        help='landmarks data file path of test data')
    parser.add_argument("--save_path", default='data/', type=str,
                        help='the save path of the data')
    parser.add_argument("--k", default=8, type=int,
                        help='k fold cross validation')
    args = parser.parse_args()

    train_path = args.train_path
    test_path = args.test_path
    label_path = args.label_path
    save_path = args.save_path
    train_landmarks_path = args.train_landmarks_path
    test_landmarks_path = args.test_landmarks_path
    k = args.k

    # with open(train_landmarks_path, 'rb') as f:
    #     train_landmarks = pickle.load(f)
    # test_dir_path = 'xin_data/train_dataset/lip_train/0ed750e9a99cb0e2331fab89e1955597'
    # imgs = read_asample_imgs(test_dir_path)
    # clip_mouth(imgs=imgs, lms=train_landmarks[test_dir_path])

    save_vocab_path = os.path.join(save_path, 'vocab.txt')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    #######################################
    #             解析并保存词表
    #######################################
    id2word, word2label = get_vocab(label_path)
    save_vocab(word2label, save_vocab_path)

    #######################################
    #            解析并保存训练数据
    #######################################
    with open(train_landmarks_path, 'rb') as f:
        train_landmarks = pickle.load(f)
    train_data, train_labels = read_data(train_landmarks, id2word, word2label)
    # 分割成k折
    train_ids, eval_ids = split_k_flod(len(train_data) // enhance_num_per_sample, k)
    for fold_i in range(k):
        # 保存每一折数据
        save_flod_data(train_data, train_labels, train_ids[fold_i], eval_ids[fold_i],
                       save_path=os.path.join(save_path, 'data_fold_{}.dat'.format(fold_i)))
    print('训练数据已保存.')


    #######################################
    #           解析并保存测试数据
    #######################################
    with open(test_landmarks_path, 'rb') as f:
        test_landmarks = pickle.load(f)
    test_data, test_ids = read_data(test_landmarks, data_type='test')
    print('shuffle test data...')
    test_data, test_ids = shuffle_data(test_data, test_ids)
    with open(os.path.join(save_path, 'test_data.dat'), 'wb') as f:
        pickle.dump(test_data, f)
        pickle.dump(test_ids, f)
    print('测试数据已保存.')

