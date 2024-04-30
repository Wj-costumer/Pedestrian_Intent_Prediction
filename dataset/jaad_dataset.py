from torch.utils.data import DataLoader, Dataset 
from .jaad_data import JAAD
import numpy as np
import os
import pickle
import tqdm
import yaml
import cv2
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, ToTensor
from torchvision.transforms.functional import crop
import torch.functional as F 

class LoopPadding(object):
    '''
    序列填充：当序列长度小于设定的最小长度时，循环填充最后一个 index 至长度等于设定的最小长度
    '''
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out

class SlidingWindow(object):
    '''
    滑动窗口：根据子序列长度和滑动步长，对一个长序列进行采样，组成多个长度相等的子序列
    '''
    def __init__(self, size, stride=0):
        self.size = size
        if stride == 0:
            self.stride = self.size
        else:
            self.stride = stride
        self.loop = LoopPadding(size)

    def __call__(self, frame_indices):
        out = []
        for begin_index in frame_indices[::self.stride]:
            end_index = min(frame_indices[-1] + 1, begin_index + self.size)
            sample = list(range(begin_index, end_index))

            if len(sample) < self.size:
                out.append(self.loop(sample))
                break
            else:
                out.append(sample)

        return out

def get_path(file_name='',
             sub_folder='',
             save_folder='models',
             dataset='pie',
             save_root_folder='data/'):
    """
    Generates paths for saving model and config data.
    Args:
        file_name: The actual save file name , e.g. 'model.h5'
        sub_folder: If another folder to be created within the root folder
        save_folder: The name of folder containing the saved files
        dataset: The name of the dataset used
        save_root_folder: The root folder
    Return:
        The full path and the path to save folder
    """
    save_path = os.path.join(save_root_folder, dataset, save_folder, sub_folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return os.path.join(save_path, file_name), save_path

sample_size = 112
mean = [0.4345, 0.4051, 0.3775]
std = [0.2768, 0.2713, 0.2737]


class JAADDataset(Dataset):
    def __init__(self, data_root, train=True, dataset='jaad', configs=None, transform=None):
        super(JAADDataset, self).__init__()
        self.data_root = data_root
        self.train = train
        self.dataset = dataset
        if transform is None:
            self.transform = Compose([
                                Resize(sample_size),
                                ToTensor(),
                                Normalize(mean, std),
                                lambda x: x[None, ...]
                            ])
        else:
            self.transform = transform
        self.test_temporal_transforms = SlidingWindow(16, 16)
        if dataset == 'jaad':
            self.imdb = JAAD(data_path=data_root)
        elif dataset == 'pie':
            pass
        # get sequences
        if self.train:
            self.beh_seq_train = self.imdb.generate_data_trajectory_sequence('train', configs['data_opts'])
            self.data = self.get_data_sequence(self.train, self.beh_seq_train, configs['data_opts'])
        else:
            self.beh_seq_test = self.imdb.generate_data_trajectory_sequence('test', configs['data_opts']) 
            self.data = self.get_data_sequence(self.train, self.beh_seq_test, configs['data_opts'])
        
    def __getitem__(self, index):
        imgs_path = self.data['image'][index]
        center = self.data['center'][index]
        frames = [Image.fromarray(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)) for img_path in imgs_path]
        frames_transform = []
        for i in range(len(frames)):
            left, top = (center[i][0] - sample_size // 2), (center[i][1] - sample_size // 2)
            frame = crop(frames[i], top, left, sample_size, sample_size)
            frames_transform.append(self.transform(frame).squeeze(0)) # torch.Size([1, 3, 112, 112])
        frames_transform = torch.stack(frames_transform, 0) # torch.Size([16, 3, 112, 112])
        return frames_transform, self.data['distance'][index], self.data['crossing'][index]

    def __len__(self):
        return len(self.data['ped_id']) 

    def get_data_sequence(self, data_type, data_raw, opts):
        """
        Generates raw sequences from a given dataset
        Args:
            data_type: Split type of data, whether it is train, test or val
            data_raw: Raw tracks from the dataset
            opts:  Options for generating data samples
        Returns:
            A list of data samples extracted from raw data
            Positive and negative data counts
        """
        print(f'Generating raw {data_type} data!')
        d = {'center': data_raw['center'].copy(),
             'box': data_raw['bbox'].copy(),
             'ped_id': data_raw['pid'].copy(),
             'crossing': data_raw['activities'].copy(),
             'image': data_raw['image'].copy(),
             'traffic': data_raw['traffic'].copy()
             }

        balance = opts['balance_data'] if data_type == 'train' else False
        obs_length = opts['obs_length']
        time_to_event = opts['time_to_event']
        normalize = opts['normalize_boxes']

        try:
            d['speed'] = data_raw['obd_speed'].copy()
        except KeyError:
            d['speed'] = data_raw['vehicle_act'].copy()
            print('Jaad dataset does not have speed information')
            print('Vehicle actions are used instead')
        if balance:
            self.balance_data_samples(d, data_raw['image_dimension'][0])
        d['box_org'] = d['box'].copy()

        if isinstance(time_to_event, int):
            for k in d.keys():
                for i in range(len(d[k])):
                    d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
            d['tte'] = [[time_to_event]]*len(data_raw['bbox'])
        else:
            overlap = opts['overlap'] # if data_type == 'train' else 0.0
            olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
            olap_res = 1 if olap_res < 1 else olap_res
            for k in d.keys():
                seqs = []
                for seq in d[k]: # a video
                    for indexs in self.test_temporal_transforms(range(len(seq)))[:2]: # 前32帧
                        sample = np.array(seq)[indexs]
                        assert len(sample) == 16 
                        seqs.append(sample)
                    
                d[k] = seqs

        # calculate the distance from the center of the image to the center of the box
        img_w = data_raw['image_dimension'][0]
        img_h = data_raw['image_dimension'][1]
        dist = []
        for i in range(len(d['center'])):
            center = d['center'][i]
            box = d['box'][i]
            speed_w = d['speed'][i] + 1
            traffic = d['traffic'][i].tolist()
            traffic_w = [[t[0]['ped_crossing'] + t[0]['ped_sign'] + t[0]['stop_sign'] + 1] for t in traffic]
            traffic_w = np.array(traffic_w)
            dist_o = self.calc_dist(center, box, img_w, img_h) * speed_w * traffic_w
            dist.append(dist_o)
        
        d['distance'] = dist
        d['crossing'] = np.array(d['crossing'])[:, 0, :]
        pos_count = np.count_nonzero(d['crossing'])
        neg_count = len(d['crossing']) - pos_count
        print("Negative {} and positive {} sample counts".format(neg_count, pos_count))

        return d

    def calc_dist(self, center, box, img_w, img_h):
        if isinstance(center, list):
            dist_x = abs(center[0] - img_w/2)/(box[2] - box[0])
            dist_y = abs(center[1] - img_h)/(box[3] - box[1])
            if dist_x == 0:
                dist_x += 1e-6
            if dist_y == 0:
                dist_y += 1e-6
            return [dist_x, dist_y]
        else:
            dist_x = abs(center[:, 0] - img_w/2)/(box[:, 2] - box[:, 0])
            dist_y = abs(center[:, 1] - img_h)/(box[:, 3] - box[:, 1])
            return np.array([dist_x, dist_y]).T
    
    def balance_data_samples(self, d, img_width, balance_tag='crossing'):
        """
        Balances the ratio of positive and negative data samples. The less represented
        data type is augmented by flipping the sequences
        Args:
            d: Sequence of data samples
            img_width: Width of the images
            balance_tag: The tag to balance the data based on
        """
        print("Balancing with respect to {} tag".format(balance_tag))
        gt_labels = [gt[0] for gt in d[balance_tag]]
        num_pos_samples = np.count_nonzero(np.array(gt_labels))
        num_neg_samples = len(gt_labels) - num_pos_samples

        # finds the indices of the samples with larger quantity
        if num_neg_samples == num_pos_samples:
            print('Positive and negative samples are already balanced')
        else:
            print('Unbalanced: \t Positive: {} \t Negative: {}'.format(num_pos_samples, num_neg_samples))
            if num_neg_samples > num_pos_samples:
                gt_augment = 1
            else:
                gt_augment = 0

            num_samples = len(d[balance_tag])
            for i in range(num_samples):
                if d[balance_tag][i][0][0] == gt_augment:
                    for k in d:
                        if k == 'center':
                            flipped = d[k][i].copy()
                            flipped = [[img_width - c[0], c[1]]
                                       for c in flipped]
                            d[k].append(flipped)
                        if k == 'box':
                            flipped = d[k][i].copy()
                            flipped = [np.array([img_width - b[2], b[1], img_width - b[0], b[3]])
                                       for b in flipped]
                            d[k].append(flipped)
                        if k == 'image':
                            flipped = d[k][i].copy()
                            flipped = [im.replace('.png', '_flip.png') for im in flipped]
                            d[k].append(flipped)
                        if k in ['speed', 'ped_id', 'crossing', 'walking', 'looking']:
                            d[k].append(d[k][i].copy())

            gt_labels = [gt[0] for gt in d[balance_tag]]
            num_pos_samples = np.count_nonzero(np.array(gt_labels))
            num_neg_samples = len(gt_labels) - num_pos_samples
            if num_neg_samples > num_pos_samples:
                rm_index = np.where(np.array(gt_labels) == 0)[0]
            else:
                rm_index = np.where(np.array(gt_labels) == 1)[0]

            # Calculate the difference of sample counts
            dif_samples = abs(num_neg_samples - num_pos_samples)
            # shuffle the indices
            np.random.seed(42)
            np.random.shuffle(rm_index)
            # reduce the number of indices to the difference
            rm_index = rm_index[0:dif_samples]

            # update the data
            for k in d:
                seq_data_k = d[k]
                d[k] = [seq_data_k[i] for i in range(0, len(seq_data_k)) if i not in rm_index]

            new_gt_labels = [gt[0] for gt in d[balance_tag]]
            num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
            print('Balanced:\t Positive: %d  \t Negative: %d\n'
                  % (num_pos_samples, len(d[balance_tag]) - num_pos_samples))



    