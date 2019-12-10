import os
import pickle
import argparse
from tqdm import tqdm
import face_alignment
from multiprocessing import Process

def check_available_fname(fname):
    try:
        if fname.split('.')[0].isdigit():
            return True
        else:
            return False
    except:
        return False

def pad_no_face_lankmark(lms, none_idx):
    if not lms:
        return None

    left_lm = None
    for i in range(none_idx, -1, -1):
        if lms[i] is not None:
            left_lm = lms[i]
            break

    rigth_lm = None
    for i in range(none_idx, len(lms)):
        if lms[i] is not None:
            rigth_lm = lms[i]
            break

    if left_lm is None and rigth_lm is None:
        return None
    left_lm = rigth_lm if left_lm is None else left_lm
    rigth_lm = left_lm if rigth_lm is None else rigth_lm
    lms[none_idx] = (left_lm + rigth_lm) / 2
    return lms


def logging(s, log_path='logs/landmark_log.txt', print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+', encoding='utf-8') as f_log:
            f_log.write(s + '\n')


def get_and_save_landmark(rank, data_dirs, landmark_to_save):
    device = 'cuda:{}'.format(rank+1)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
    if rank == 0:
        bar = tqdm(range(len(data_dirs)))
    p_landmarks = {}

    for img_dir in data_dirs:
        if rank == 0:bar.update(1)
        ################################
        #         获取图片路径列表
        ################################
        lms = fa.get_landmarks_from_directory(img_dir, show_progress_bar=False)
        available_keys = [int(key.split('/')[-1].split('.')[0]) for key in lms.keys() if check_available_fname(key.split('/')[-1])]
        available_keys = sorted(available_keys)
        available_keys = [os.path.join(img_dir, '{}.png'.format(i)) for i in available_keys]
        if len(available_keys) != len(lms.keys()):
            logging('图片文件名有误：{}'.format(img_dir))

        ###################################
        # 筛选掉没有人脸的图像，并排序加入list
        ###################################
        none_flag = True
        clean_lms = []
        for key in available_keys:
            if not lms[key]:
                logging('没有检测到人脸：{}'.format(key))
                clean_lms.append(None)
            else:
                none_flag=False
                clean_lms.append(lms[key][0])

        if none_flag:
            logging('丢弃图片:{}'.format(img_dir))
            continue

        # 填充没有识别到人脸的图像
        for i in range(len(clean_lms)):
            if clean_lms[i] is None:
                clean_lms = pad_no_face_lankmark(clean_lms, i)

        p_landmarks[img_dir] = clean_lms

    if rank == 0:bar.close()
    with open(landmark_to_save, 'wb') as f:
        pickle.dump(p_landmarks, f)
    logging('已完成，保存至:{}'.format(landmark_to_save))
    return p_landmarks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default='hack_data/train/hack_lip_train/', type=str,
                        help='lip_train folder path')
    parser.add_argument("--save_path", default='data/train_landmarks.dat', type=str,
                        help='the landmark data path to save')
    args = parser.parse_args()

    root_dir = args.root_dir
    save_path = args.save_path
    temp_dir = 'temp/'
    word_size = 2
    dir_list = [os.path.join(root_dir, d) for d in os.listdir(root_dir)]
    num_dirs_per_process = len(dir_list) // word_size

    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    ps = []
    for rank in range(word_size):
        end_idx = (rank + 1) * num_dirs_per_process if rank != word_size - 1 else len(dir_list)
        data_dirs = dir_list[rank * num_dirs_per_process:end_idx]
        landmark_to_save = os.path.join(temp_dir, '{}.tmp'.format(rank))
        device = 'cuda:{}'.format(rank)
        p = Process(target=get_and_save_landmark,
                    args=(rank, data_dirs, landmark_to_save))
        ps.append(p)
    for i in range(word_size):
        ps[i].start()
    for i in range(word_size):
        ps[i].join()

    landmarks = {}
    for i in range(word_size):
        with open(os.path.join(temp_dir, '{}.tmp'.format(i)), 'rb') as f:
            landmarks.update(pickle.load(f))
    with open(save_path, 'wb') as f:
        pickle.dump(landmarks, f)
