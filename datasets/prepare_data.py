# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/PerAda/blob/main/LICENSE

from torchvision import datasets, transforms
import os 
import numpy as np
import torch
import sys

def _get_transform(data_name, is_training,img_resolution=None):
    if data_name == "cifar10" or data_name == "cifar10.1" or "CIFAR-10-C" in data_name:
        img_resolution=  img_resolution if img_resolution is not None else 32

        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
        )
        if is_training:
            transform = transforms.Compose(
                [   
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop((32, 32), 4),
                    transforms.Resize(img_resolution),
                    transforms.ToTensor(), 
                    normalize,
                ]
            )
        else:
            transform = transforms.Compose([transforms.Resize(img_resolution),transforms.ToTensor(),normalize])

    elif data_name == "cifar100" or data_name == "cifar100-lt":
        normalize = transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        )
        img_resolution=  img_resolution if img_resolution is not None else 32
        if is_training:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop((32, 32), 4),
                    transforms.Resize(img_resolution),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform = transforms.Compose([transforms.Resize(img_resolution),transforms.ToTensor(),normalize])

    elif data_name == "stl10":
        normalize = transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        )
        img_resolution=  img_resolution if img_resolution is not None else 96
        if is_training:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop((96, 96), 4),
                    transforms.Resize((img_resolution, img_resolution)),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform = transforms.Compose([
                transforms.Resize((img_resolution, img_resolution)),
                transforms.ToTensor(),
                normalize
                ])
            
    elif data_name == "inat2018":
        normalize = transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        )
        img_resolution=  img_resolution if img_resolution is not None else 224
        if is_training:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    # transforms.RandomCrop((224, 224), 4),
                    transforms.Resize((img_resolution, img_resolution)),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform = transforms.Compose([
                transforms.Resize((img_resolution, img_resolution)),
                transforms.ToTensor(),
                normalize
                ])
            
    else:
        raise NotImplementedError


    print("img_resolution", img_resolution)

    return transform


def _get_cifar(data_name, root, split, transform, target_transform, download):
    is_train = split == "train"
    # decide normalize parameter.
    
    if data_name == "cifar10":
        dataset_loader = datasets.CIFAR10(root=root,
                train=is_train,
                transform=transform,
                target_transform=target_transform,
                download=download,
            )
    elif data_name == "cifar100":
        dataset_loader = datasets.CIFAR100(root=root,
                train=is_train,
                transform=transform,
                target_transform=target_transform,
                download=download,
            )
    elif data_name == "cifar100-lt":
        dataset = datasets.CIFAR100(root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
            )
        # dataset = _make_cifar100_lt(dataset, imb_factor=0.01)

        dataset_loader = dataset
        
    elif data_name == "cifar10.1":
        from datasets.cifar import CIFAR101
        dataset_loader = CIFAR101(
                    root,
                     transform=transform
                )
    elif "CIFAR-10-C" in data_name:
        cname = data_name.split("@")[1]
        from datasets.cifar import CIFAR10C
        dataset_loader = CIFAR10C(
                   os.path.join('data', 'CIFAR-10-C'),  cname,  # hardcode the path..
                     transform=transform
                )
    else:
        raise NotImplementedError(f"invalid data_name={data_name}.")


    return dataset_loader



def _get_stl10(data_name, root, split, transform, target_transform, download):
    # right now this function is only used for unlabeled dataset.
    is_train = split == "train"

    if is_train:
        split = "train+unlabeled" # 105000 data

    return datasets.STL10(
        root=root,
        split=split,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )
    

def _get_inat2018(split, transform, target_transform, download, allowed_category_ids=None):

    root = '/mnt/dataset/opendata/INaturalist'
    image_root = os.path.join(root, "train_val2018")

    train_dataset = INat2018Dataset(
            root=root,
            json_path=root+'/train2018.json',
            transform=transform,
            min_samples_per_class=40,
        )
    
    train_label_counts = Counter(train_dataset.labels)
    valid_class_indices = [label for label, count in train_label_counts.items() if count >= 40]
    valid_category_ids = [train_dataset.index_to_label[idx] for idx in valid_class_indices]
    
    if split == 'train':
        return train_dataset
    else:
        return INat2018Dataset(
            root=root,
            json_path=root+'/val2018.json',
            transform=transform,
            allowed_category_ids=valid_category_ids
        )


def get_dataset(
    data_name,
    datasets_path,
    split="train",
    transform=None,
    target_transform=None,
    download=True,
    img_resolution=None,
):
    # create data folder if it does not exist.
    root = os.path.join(datasets_path, data_name)
    is_train = split == "train"
    transform = _get_transform(data_name=data_name, is_training= is_train, img_resolution= img_resolution)
    if data_name == "cifar10" or data_name == "cifar100" or data_name == "cifar100-lt" or data_name == "cifar10.1" or "CIFAR-10-C" in data_name:
        return _get_cifar(data_name, root, split, transform, target_transform, download)
    elif data_name == "stl10":
        return _get_stl10(data_name, root, split, transform, target_transform, download)
    elif data_name == "inat2018":
        return _get_inat2018(split, transform, target_transform, download)
    
    else: 
        raise NotImplementedError


def cifar_noniid(dataset, num_users, user_split=1, alpha =None, shard_per_user=None, proportions_dict=None):
    """
    Sample non-I.I.D client data from CIFAR dataset
    :param dataset:
    :param num_users:
    :return:
    """
    
    total_number_samples= len(dataset)
    num_classes = len(np.unique(dataset.targets))
   

    num_user_data = int( user_split * total_number_samples)
    labels = np.array(dataset.targets)
    _lst_sample = 2
    if alpha is not None:
        method= 'dir'
    elif shard_per_user is not None:
        method= 'shard'
    
    print("cifar_noniid", total_number_samples,num_classes,method) 
    
    
    if method=="shard":
        dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
        idxs_dict = {}
        server_idx = np.random.choice(list(set(range(total_number_samples))) , total_number_samples- num_user_data , replace=False)
        local_idx = [i for i in range(total_number_samples) if i not in server_idx]
        
        for i in local_idx:
            label = torch.tensor(dataset.targets[i]).item()
            if label not in idxs_dict.keys():
                idxs_dict[label] = []
            idxs_dict[label].append(i)

       
        shard_per_class = int(shard_per_user * num_users / num_classes)
        for label in idxs_dict.keys():
            x = idxs_dict[label]
            num_leftover = len(x) % shard_per_class
            leftover = x[-num_leftover:] if num_leftover > 0 else []
            x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
            x = x.reshape((shard_per_class, -1))
            x = list(x)

            for i, idx in enumerate(leftover):
                x[i] = np.concatenate([x[i], [idx]])
            idxs_dict[label] = x


        if proportions_dict is None:
            proportions_dict = list(range(num_classes)) * shard_per_class
            np.random.shuffle(proportions_dict)
            proportions_dict = np.array(proportions_dict).reshape((num_users, -1))

        # Divide and assign
        for i in range(num_users):
            rand_set_label = proportions_dict[i]
            rand_set = []
            for label in rand_set_label:
                idx = np.random.choice(len(idxs_dict[label]), replace=False)
                rand_set.append(idxs_dict[label].pop(idx))
            dict_users[i] = np.concatenate(rand_set)
    
    elif method == "dir":
        if proportions_dict is None:
            proportions_dict= {k: None for k in range(num_classes)}

        y_train = labels
        
        least_idx = np.zeros((num_users, num_classes, _lst_sample), dtype=int)
        for i in range(num_classes):
            idx_i = np.random.choice(np.where(labels==i)[0], num_users*_lst_sample, replace=False)
            least_idx[:, i, :] = idx_i.reshape((num_users, _lst_sample))
        least_idx = np.reshape(least_idx, (num_users, -1))
        
        least_idx_set = set(np.reshape(least_idx, (-1)))
        #least_idx_set = set([])
        
        server_idx = np.random.choice(list(set(range(total_number_samples))-least_idx_set), total_number_samples-num_user_data, replace=False)
        local_idx = np.array([i for i in range(total_number_samples) if i not in server_idx and i not in least_idx_set])
    
        print(len(server_idx), len(local_idx), len(least_idx_set) )
        N = y_train.shape[0]
        net_dataidx_map = {}
        dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
        
        idx_batch = [[] for _ in range(num_users)]
        # for each class in the dataset
        for k in range(num_classes):
            idx_k_select = np.where(y_train == k)[0]
           
            idx_k =[]
            for id in idx_k_select:
                if id in local_idx:
                    idx_k.append(id)

            np.random.shuffle(idx_k)
            if proportions_dict[k] is not None:
                proportions = proportions_dict[k]
            else:
                proportions = np.random.dirichlet(np.repeat(alpha, num_users))
                proportions_dict[k] = proportions
            ## Balance
            proportions = np.array([p*(len(idx_j)<N/num_users) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
           

        for j in range(num_users):
            np.random.shuffle(idx_batch[j])
            dict_users[j] = idx_batch[j]  
            dict_users[j] = np.concatenate((dict_users[j], least_idx[j]), axis=0)    
                  

    # print(proportions_dict)
    cnts_dict = {}
    client_losses = {}

    # with open("data_%d_u%d_%s.txt"%(num_data, num_users, method), 'w') as f:
    # for i in range(num_users):
    #     dict_users[i] = [int(index) for index in dict_users[i]]
    #     labels_i = labels[dict_users[i]]
    #     cnts = np.array([np.count_nonzero(labels_i == j ) for j in range(num_classes)] )
    #     cnts_dict[i] = cnts
    #     # f.write("User %s: %s sum: %d\n"%(i, " ".join([str(cnt) for cnt in cnts]), sum(cnts) ))  
    #     # print("User %s: %s sum: %d\n"%(i, " ".join([str(cnt) for cnt in cnts]), sum(cnts) ))
    #     ## Dynamic loss code
    #     imbalance_ratio, is_long_tail, class_counts = compute_imbalance_ratio(labels_i, num_classes)  
    #     client_losses[i] = DynamicLoss(class_counts=class_counts)
    #     ##cls_num code

        

    for i in range(num_users):
        dict_users[i]= list(dict_users[i])
    server_idx =list(server_idx)
    return dict_users, server_idx, cnts_dict, proportions_dict, client_losses




def svhn_noniid(dataset, num_users, user_split=1, alpha =None, shard_per_user=None, proportions_dict=None):
    """
    Sample non-I.I.D client data from SVHN dataset
    :param dataset:
    :param num_users:
    :return:
    """
    
    total_number_samples= len(dataset)
    num_classes = len(np.unique(dataset.labels))
   

    num_user_data = int( user_split * total_number_samples)
    labels = np.array(dataset.labels)
    _lst_sample = 2
    if alpha is not None:
        method= 'dir'
    elif shard_per_user is not None:
        method= 'shard'
    
    print("svhn_noniid", total_number_samples,num_classes,method) 
    
    
    if method=="shard":
        dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
        idxs_dict = {}
        server_idx = np.random.choice(list(set(range(total_number_samples))) , total_number_samples- num_user_data , replace=False)
        local_idx = [i for i in range(total_number_samples) if i not in server_idx]
        
        for i in local_idx:
            label = torch.tensor(dataset.labels[i]).item()
            if label not in idxs_dict.keys():
                idxs_dict[label] = []
            idxs_dict[label].append(i)

       
        shard_per_class = int(shard_per_user * num_users / num_classes)
        for label in idxs_dict.keys():
            x = idxs_dict[label]
            num_leftover = len(x) % shard_per_class
            leftover = x[-num_leftover:] if num_leftover > 0 else []
            x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
            x = x.reshape((shard_per_class, -1))
            x = list(x)

            for i, idx in enumerate(leftover):
                x[i] = np.concatenate([x[i], [idx]])
            idxs_dict[label] = x


        if proportions_dict is None:
            proportions_dict = list(range(num_classes)) * shard_per_class
            np.random.shuffle(proportions_dict)
            proportions_dict = np.array(proportions_dict).reshape((num_users, -1))

        # Divide and assign
        for i in range(num_users):
            rand_set_label = proportions_dict[i]
            rand_set = []
            for label in rand_set_label:
                idx = np.random.choice(len(idxs_dict[label]), replace=False)
                rand_set.append(idxs_dict[label].pop(idx))
            dict_users[i] = np.concatenate(rand_set)
    
    elif method == "dir":
        if proportions_dict is None:
            proportions_dict= {k: None for k in range(num_classes)}

        y_train = labels
        
        least_idx = np.zeros((num_users, num_classes, _lst_sample), dtype=int)
        for i in range(num_classes):
            idx_i = np.random.choice(np.where(labels==i)[0], num_users*_lst_sample, replace=False)
            least_idx[:, i, :] = idx_i.reshape((num_users, _lst_sample))
        least_idx = np.reshape(least_idx, (num_users, -1))
        
        least_idx_set = set(np.reshape(least_idx, (-1)))
        #least_idx_set = set([])
        
        server_idx = np.random.choice(list(set(range(total_number_samples))-least_idx_set), total_number_samples-num_user_data, replace=False)
        local_idx = np.array([i for i in range(total_number_samples) if i not in server_idx and i not in least_idx_set])
    
        print(len(server_idx), len(local_idx), len(least_idx_set) )
        N = y_train.shape[0]
        net_dataidx_map = {}
        dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
        
        idx_batch = [[] for _ in range(num_users)]
        # for each class in the dataset
        for k in range(num_classes):
            idx_k_select = np.where(y_train == k)[0]
        
            idx_k =[]
            for id in idx_k_select:
                if id in local_idx:
                    idx_k.append(id)

            np.random.shuffle(idx_k)
            if proportions_dict[k] is not None:
                proportions = proportions_dict[k]
            else:
                proportions = np.random.dirichlet(np.repeat(alpha, num_users))
                proportions_dict[k] = proportions
            ## Balance
            proportions = np.array([p*(len(idx_j)<N/num_users) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
           

        for j in range(num_users):
            np.random.shuffle(idx_batch[j])
            dict_users[j] = idx_batch[j]  
            dict_users[j] = np.concatenate((dict_users[j], least_idx[j]), axis=0)          

    cnts_dict = {}
    
    for i in range(num_users):
        dict_users[i] = [int(index) for index in dict_users[i]]
        labels_i = labels[dict_users[i]]
        cnts = np.array([np.count_nonzero(labels_i == j ) for j in range(num_classes)] )
        cnts_dict[i] = cnts
    

    for i in range(num_users):
        dict_users[i]= list(dict_users[i])
    server_idx =list(server_idx)
    return dict_users, server_idx, cnts_dict, proportions_dict


def inat2018_noniid(dataset, num_users, user_split=1, alpha=None, shard_per_user=None, proportions_dict=None):
    labels = np.array(dataset.labels)
    
    labels_unique, labels = np.unique(labels, return_inverse=True)
    num_classes = len(labels_unique)
    
    total_number_samples = len(np.unique(labels))
    num_user_data = int(user_split * total_number_samples)
    _lst_sample = 2  # minimum guaranteed samples per class per user

    if alpha is not None:
        method = 'dir'
    elif shard_per_user is not None:
        method = 'shard'
        raise NotImplementedError("Shard-based split is not implemented for iNaturalist.")
    else:
        raise ValueError("Either alpha or shard_per_user must be specified.")

    print("inat2018_noniid", total_number_samples, num_classes, method)

    if proportions_dict is None:
        proportions_dict = {k: None for k in range(num_classes)}

    y_train = labels
    least_idx = np.zeros((num_users, num_classes, _lst_sample), dtype=int)

    for i in range(num_classes):
        idx_i = np.random.choice(np.where(labels==i)[0], num_users*_lst_sample, replace=False)
        least_idx[:, i, :] = idx_i.reshape((num_users, _lst_sample))

    least_idx = np.reshape(least_idx, (num_users, -1))

    least_idx_set = set(least_idx.flatten())
    all_indices = set(range(total_number_samples))
    server_idx = np.random.choice(list(all_indices - least_idx_set), total_number_samples - num_user_data, replace=False)
    server_idx_set = set(server_idx)
    local_idx = np.array([i for i in range(total_number_samples) if i not in server_idx_set and i not in least_idx_set])

    print("Server/Local/Least:", len(server_idx), len(local_idx), len(least_idx_set))

    N = len(y_train)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idx_batch = [[] for _ in range(num_users)]

    for k in range(num_classes):
        idx_k_select = np.where(y_train == k)[0]
        idx_k = [id for id in idx_k_select if id in local_idx]
        np.random.shuffle(idx_k)

        if proportions_dict[k] is not None:
            proportions = proportions_dict[k]
        else:
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            proportions_dict[k] = proportions

        proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = np.concatenate((idx_batch[j], least_idx[j]), axis=0)

    cnts_dict = {}
    for i in range(num_users):
        dict_users[i] = [int(index) for index in dict_users[i]]
        labels_i = labels[dict_users[i]]
        cnts = np.array([np.count_nonzero(labels_i == j) for j in range(num_classes)])
        cnts_dict[i] = cnts

    for i in range(num_users):
        dict_users[i] = list(dict_users[i])
    server_idx = list(server_idx)

    return dict_users, server_idx, cnts_dict, proportions_dict


def inat2018_noniid_soft(dataset, num_users, alpha=None, proportions_dict=None):
    labels = np.array(dataset.labels)
    labels_unique, labels = np.unique(labels, return_inverse=True)
    num_classes = len(labels_unique)

    print("inat2018_noniid_soft", len(labels), num_classes)

    if proportions_dict is None:
        proportions_dict = {k: None for k in range(num_classes)}

    dict_users = {i: [] for i in range(num_users)}
    idx_batch = [[] for _ in range(num_users)]

    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)

        if proportions_dict[k] is not None:
            proportions = proportions_dict[k]
        else:
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            proportions_dict[k] = proportions

        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_split = np.split(idx_k, proportions)

        for i, idx in enumerate(idx_split):
            idx_batch[i].extend(idx.tolist())

    # Shuffle and assign to dict
    for i in range(num_users):
        np.random.shuffle(idx_batch[i])
        dict_users[i] = idx_batch[i]

    # Count per user
    cnts_dict = {}
    for i in range(num_users):
        labels_i = labels[dict_users[i]]
        cnts = np.array([np.count_nonzero(labels_i == j) for j in range(num_classes)])
        cnts_dict[i] = cnts

    return dict_users, None, cnts_dict, proportions_dict





from torch.utils.data import Dataset
from PIL import Image
import os
import json
from collections import Counter

class INat2018Dataset(Dataset):
    def __init__(self, root, json_path, transform=None, target_transform=None,
                 min_samples_per_class=0, allowed_category_ids=None, val_json_path=None, min_val_samples=0):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.images = data['images']
        self.annotations = data['annotations']
        self.categories = data['categories']

        self.img_id_to_path = {img['id']: img['file_name'] for img in self.images}
        self.img_id_to_label = {ann['image_id']: ann['category_id'] for ann in self.annotations}

        # 전체 샘플 구성
        all_samples = [
            (self.img_id_to_path[img_id], self.img_id_to_label[img_id])
            for img_id in self.img_id_to_path
            if img_id in self.img_id_to_label
        ]

        # 기본적으로 min_samples_per_class 기준으로 class 필터링
        label_counts = Counter(label for _, label in all_samples)
        allowed_labels = {label for label, count in label_counts.items() if count >= min_samples_per_class}

        # validation 기준으로 class 필터링
        if val_json_path is not None and min_val_samples > 0:
            with open(val_json_path, 'r') as f:
                val_data = json.load(f)
            val_annotations = val_data['annotations']
            val_label_counts = Counter(ann['category_id'] for ann in val_annotations)
            val_allowed_labels = {label for label, count in val_label_counts.items() if count >= min_val_samples}
            allowed_labels &= val_allowed_labels

        # allowed_category_ids가 따로 주어진 경우에도 적용
        if allowed_category_ids is not None:
            allowed_labels &= set(allowed_category_ids)

        # 최종 샘플 필터링
        filtered_samples = [
            (img, label) for (img, label) in all_samples
            if label in allowed_labels
        ]

        # label 재매핑
        used_labels = sorted(set(label for _, label in filtered_samples))
        self.label_to_index = {label: idx for idx, label in enumerate(used_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

        self.samples = [(img, self.label_to_index[label]) for (img, label) in filtered_samples]
        self.labels = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path_rel, label = self.samples[idx]
        img_path = os.path.join(self.root, img_path_rel)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

