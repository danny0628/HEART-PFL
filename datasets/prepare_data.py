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
import glob
from collections import defaultdict
from typing import List, Tuple, Optional
from PIL import Image
from torch.utils.data import Dataset

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
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )
        img_resolution=  img_resolution if img_resolution is not None else 256
        if is_training:
            transform = transforms.Compose(
                [
                    transforms.Resize(img_resolution),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
    elif data_name == "flower102" or data_name == "flower_kd" or data_name == "flower_kd2":
        if is_training:
            transform = transforms.Compose(
                [transforms.Resize((64, 64)), 
                transforms.ToTensor(), 
                transforms.Normalize((0.5), (0.5))]
            )
        else:
            transform = transforms.Compose(
                [transforms.Resize((64, 64)), 
                transforms.ToTensor(), 
                transforms.Normalize((0.5), (0.5))]
            )
    elif data_name == "caltech101" or data_name == "caltech_kd":        
        if is_training:
            transform = transforms.Compose(
                [transforms.Resize((128, 128)),
                transforms.Lambda(lambda x: x.convert('RGB')),  # RGB로 변환 추가 
                transforms.ToTensor(), 
                transforms.Normalize(mean = [0.485,0.456,0.406], std=[0.229,0.224,0.225])
                ]
            )
        else:
            transform = transforms.Compose(
                [transforms.Resize((128, 128)), 
                transforms.Lambda(lambda x: x.convert('RGB')),  # RGB로 변환 추가
                transforms.ToTensor(), 
                transforms.Normalize(mean = [0.485,0.456,0.406], std=[0.229,0.224,0.225])
                ]
            )    
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

import torch
from torch.utils.data import ConcatDataset, Subset
from torchvision import datasets

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset
from torchvision import datasets

def _labels_array(dataset):
    """Flowers102 라벨을 np.array로 안전 추출 (targets / labels / Subset / ConcatDataset 모두 대응)"""
    # 1) 바로 속성 있는 경우
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels)

    # 2) Subset
    if isinstance(dataset, Subset):
        base = _labels_array(dataset.dataset)
        idxs = dataset.indices
        return np.array([base[i] for i in idxs])

    # 3) ConcatDataset
    if isinstance(dataset, ConcatDataset):
        parts = [_labels_array(ds) for ds in dataset.datasets]
        return np.concatenate(parts, axis=0)

    # 4) 최후 수단: __getitem__
    labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        labels.append(int(y))
    return np.array(labels)

def _get_flower102(data_name, root, split, transform, target_transform, download):
    is_train = split == "train"

    if data_name != "flower102":
        raise NotImplementedError(f"invalid data_name={data_name}.")

    # 공식 split 3개 로드 후 합치고 75/25로 서브셋
    tr = datasets.Flowers102(root=root, split="train",
                             transform=transform, target_transform=target_transform,
                             download=download)
    va = datasets.Flowers102(root=root, split="val",
                             transform=transform, target_transform=target_transform,
                             download=False)
    te = datasets.Flowers102(root=root, split="test",
                             transform=transform, target_transform=target_transform,
                             download=False)

    merged = ConcatDataset([tr, va, te])

    all_labels = _labels_array(merged)
    all_labels = _normalize_labels_zero_based(all_labels)

    num_classes = int(all_labels.max() + 1)
    g = torch.Generator().manual_seed(1)

    train_idx, val_idx = [], []
    for c in range(num_classes):
        cls_idx = np.where(all_labels == c)[0]
        # 섞기 (재현성)
        perm = torch.randperm(len(cls_idx), generator=g).numpy()
        cls_idx = cls_idx[perm]
        n_train = int(round(0.75 * len(cls_idx)))
        train_idx.append(cls_idx[:n_train])
        val_idx.append(cls_idx[n_train:])

    train_idx = np.concatenate(train_idx).tolist()
    val_idx   = np.concatenate(val_idx).tolist()

    return Subset(merged, train_idx) if is_train else Subset(merged, val_idx)

def _get_caltech101(data_name, root, split, transform, target_transform, download):
    is_train = split == "train"

    if data_name != "caltech101":
        raise NotImplementedError(f"invalid data_name={data_name}.")

    full_dataset = datasets.Caltech101(
        root='/data', 
        transform=transform, 
        target_transform=target_transform,
        download=False
    )

    all_labels = _labels_array(full_dataset)
    all_labels = _normalize_labels_zero_based(all_labels)
    
    num_classes = int(all_labels.max() + 1)
    g = torch.Generator().manual_seed(1)
    
    train_idx, val_idx = [], []
    for c in range(num_classes):
        cls_idx = np.where(all_labels == c)[0]
        if len(cls_idx) == 0:
            continue
            
        perm = torch.randperm(len(cls_idx), generator=g).numpy()
        cls_idx = cls_idx[perm]
        
        n_train = int(round(0.75 * len(cls_idx)))
        train_idx.append(cls_idx[:n_train])
        val_idx.append(cls_idx[n_train:])
    
    train_idx = np.concatenate(train_idx).tolist()
    val_idx = np.concatenate(val_idx).tolist()
    
    return Subset(full_dataset, train_idx) if is_train else Subset(full_dataset, val_idx)

def _get_flower_kd(data_name, root, split, transform, target_transform, download, num_classes_to_use=100):
    is_train = split == "train"

    if data_name != "flower_kd":
        raise NotImplementedError(f"invalid data_name={data_name}.")

    tr = datasets.Flowers102(root=root, split="train",
                             transform=transform, target_transform=target_transform,
                             download=download)
    va = datasets.Flowers102(root=root, split="val",
                             transform=transform, target_transform=target_transform,
                             download=False)
    te = datasets.Flowers102(root=root, split="test",
                             transform=transform, target_transform=target_transform,
                             download=False)

    merged = ConcatDataset([tr, va, te])
    all_labels = _labels_array(merged)
    all_labels = _normalize_labels_zero_based(all_labels)
    
    if num_classes_to_use < len(np.unique(all_labels)):
        # 첫 번째 num_classes_to_use개 클래스만 선택 (0부터 num_classes_to_use-1까지)
        selected_classes = list(range(num_classes_to_use))  # [0, 1, 2, ..., num_classes_to_use-1]
        
        # 선택된 클래스에 해당하는 샘플들만 필터링
        mask = np.isin(all_labels, selected_classes)
        filtered_indices = np.where(mask)[0]
        
        # 필터링된 레이블들 (아직 재매핑 전)
        filtered_labels = all_labels[filtered_indices]
        
        # 필터링된 데이터셋 생성 (레이블은 그대로 유지, 0~100)
        filtered_dataset = Subset(merged, filtered_indices.tolist())
        working_labels = filtered_labels
        working_dataset = filtered_dataset
        
    else:
        working_labels = all_labels
        working_dataset = merged

    num_classes = num_classes_to_use
    g = torch.Generator().manual_seed(1)

    train_idx, val_idx = [], []
    for c in range(num_classes):
        cls_idx = np.where(working_labels == c)[0]
        if len(cls_idx) == 0:
            continue
            
        perm = torch.randperm(len(cls_idx), generator=g).numpy()
        cls_idx = cls_idx[perm]
        n_train = int(round(0.75 * len(cls_idx)))
        train_idx.append(cls_idx[:n_train])
        val_idx.append(cls_idx[n_train:])

    train_idx = np.concatenate(train_idx).tolist()
    val_idx = np.concatenate(val_idx).tolist()

    final_dataset = Subset(working_dataset, train_idx) if is_train else Subset(working_dataset, val_idx)
    
    return final_dataset

# 헬퍼 함수들
def _labels_array(dataset):
    """데이터셋에서 레이블 배열 추출"""
    if hasattr(dataset, 'targets'):
        return np.array(dataset.targets)
    else:
        # ConcatDataset의 경우
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
        return np.array(labels)

def _normalize_labels_zero_based(labels):
    """레이블을 0부터 시작하도록 정규화"""
    unique_labels = np.unique(labels)
    if unique_labels[0] == 0 and len(unique_labels) == (unique_labels[-1] + 1):
        # 이미 0부터 연속된 레이블
        return labels
    else:
        # 재매핑 필요
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        return np.array([label_map[label] for label in labels])
    
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
    # elif data_name == "stl10":
    #     return _get_stl10(data_name, root, split, transform, target_transform, download)
    elif data_name == "flower102":
        return _get_flower102(data_name, root, split, transform, target_transform, download)
    elif data_name == "caltech101":
        return _get_caltech101(data_name, root, split, transform, target_transform, download)
    elif data_name == "flower_kd":
        return _get_flower_kd(data_name, root, split, transform, target_transform, download, num_classes_to_use=101)
    
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

# datasets/prepare_data.py (또는 너가 쓰는 유틸 파일)에 추가
import numpy as np
import torch

from torch.utils.data import Subset, ConcatDataset

def flower102_noniid(dataset, num_users, user_split=1, alpha=None, shard_per_user=None, proportions_dict=None):
    # ── Flower102 전용: 넘어오는 dataset은 Subset(ConcatDataset(...)) 형태일 수 있음
    def _flower102_labels(ds):
        # 1) 원본 Flowers102 속성
        if hasattr(ds, "targets"):
            arr = np.array(ds.targets)
        elif hasattr(ds, "labels"):
            arr = np.array(ds.labels)
        # 2) Subset: base에서 indices로 슬라이싱
        elif isinstance(ds, Subset):
            base = _flower102_labels(ds.dataset)
            idxs = np.array(ds.indices, dtype=np.int64)
            arr = base[idxs]
        # 3) ConcatDataset: child 들 라벨 이어붙이기
        elif isinstance(ds, ConcatDataset):
            parts = [_flower102_labels(child) for child in ds.datasets]
            arr = np.concatenate(parts, axis=0)
        # 4) 최후수단: __getitem__
        else:
            arr = np.array([int(ds[i][1]) for i in range(len(ds))], dtype=np.int64)

        # Flowers102가 1..102 라벨이면 0..101로 정규화
        if arr.min() == 1 and arr.max() >= 100:
            arr = arr - 1
        return arr.astype(np.int64, copy=False)

    total_number_samples = len(dataset)
    labels = _flower102_labels(dataset)
    num_classes = len(np.unique(labels))

    num_user_data = int(user_split * total_number_samples)
    _lst_sample = 2

    if alpha is not None:
        method = 'dir'
    elif shard_per_user is not None:
        method = 'shard'
    else:
        method = 'dir'
        alpha = 0.5 if alpha is None else alpha

    print("flower102_noniid", total_number_samples, num_classes, method)

    if method == "shard":
        dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
        idxs_dict = {}
        server_idx = np.random.choice(list(set(range(total_number_samples))),
                                      total_number_samples - num_user_data, replace=False)
        local_idx = [i for i in range(total_number_samples) if i not in server_idx]

        for i in local_idx:
            y = int(labels[i])
            idxs_dict.setdefault(y, []).append(i)

        shard_per_class = int(shard_per_user * num_users / num_classes)
        for y in idxs_dict.keys():
            x = idxs_dict[y]
            num_leftover = len(x) % shard_per_class
            leftover = x[-num_leftover:] if num_leftover > 0 else []
            x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
            x = list(x.reshape((shard_per_class, -1)))
            for i, idx in enumerate(leftover):
                x[i] = np.concatenate([x[i], [idx]])
            idxs_dict[y] = x

        if proportions_dict is None:
            proportions_dict = list(range(num_classes)) * shard_per_class
            np.random.shuffle(proportions_dict)
            proportions_dict = np.array(proportions_dict).reshape((num_users, -1))

        for i in range(num_users):
            rand_set_label = proportions_dict[i]
            rand_set = []
            for y in rand_set_label:
                j = np.random.choice(len(idxs_dict[y]), replace=False)
                rand_set.append(idxs_dict[y].pop(j))
            dict_users[i] = np.concatenate(rand_set)

    elif method == "dir":
        if proportions_dict is None:
            proportions_dict = {k: None for k in range(num_classes)}

        least_lists = [[] for _ in range(num_users)]
        for k in range(num_classes):
            cls_all = np.where(labels == k)[0]
            cls_all = np.random.permutation(cls_all)  # 섞기

            need = num_users * _lst_sample
            if len(cls_all) >= need:
                # 충분하면 정확히 _lst_sample씩 나눠줌
                pick = cls_all[:need].reshape(num_users, _lst_sample)
                for u in range(num_users):
                    least_lists[u].extend(pick[u].tolist())
            else:
                # 부족하면 가능한 만큼을 라운드로빈으로 분배
                for i, idx in enumerate(cls_all):
                    least_lists[i % num_users].append(int(idx))

        least_idx = [np.array(lst, dtype='int64') for lst in least_lists]
        least_idx_set = set(int(x) for lst in least_idx for x in lst)

        server_idx = np.random.choice(list(set(range(total_number_samples)) - least_idx_set),
                                    total_number_samples - num_user_data, replace=False)
        local_idx = np.array([i for i in range(total_number_samples)
                            if i not in server_idx and i not in least_idx_set])

        print(len(server_idx), len(local_idx), len(least_idx_set))
        N = labels.shape[0]
        dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
        idx_batch = [[] for _ in range(num_users)]

        for k in range(num_users if False else num_classes):  # 그대로 유지
            idx_k_select = np.where(labels == k)[0]
            idx_k = [i for i in idx_k_select if i in local_idx]
            np.random.shuffle(idx_k)

            if proportions_dict[k] is not None:
                proportions = proportions_dict[k]
            else:
                proportions = np.random.dirichlet(np.repeat(alpha, num_users))
                proportions_dict[k] = proportions

            proportions = np.array([p * (len(x) < N / num_users) for p, x in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            cuts = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            splits = np.split(idx_k, cuts)
            idx_batch = [x + s.tolist() for x, s in zip(idx_batch, splits)]

        for j in range(num_users):
            np.random.shuffle(idx_batch[j])
            dict_users[j] = np.array(idx_batch[j], dtype='int64')
            if len(least_idx[j]) > 0:
                dict_users[j] = np.concatenate((dict_users[j], least_idx[j]), axis=0)

    # 반환 형식 CIFAR와 동일
    for i in range(num_users):
        dict_users[i] = list(dict_users[i])
    server_idx = list(server_idx)
    cnts_dict = {}
    return dict_users, server_idx, cnts_dict, proportions_dict


def caltech101_noniid(dataset, num_users, user_split=1, alpha=None, shard_per_user=None, proportions_dict=None):
    """Caltech-101용 Non-IID 데이터 분할 함수 (최소 샘플 보장 없음)"""
    
    def _caltech101_labels(ds):
        """Caltech-101 라벨을 안전하게 추출"""
        # 1) 직접 속성 확인
        if hasattr(ds, "y"):  # Caltech-101은 y 속성 사용
            arr = np.array(ds.y)
        elif hasattr(ds, "targets"):
            arr = np.array(ds.targets)
        elif hasattr(ds, "labels"):
            arr = np.array(ds.labels)
        # 2) Subset 처리
        elif isinstance(ds, Subset):
            base = _caltech101_labels(ds.dataset)
            idxs = np.array(ds.indices, dtype=np.int64)
            arr = base[idxs]
        # 3) ConcatDataset 처리
        elif isinstance(ds, ConcatDataset):
            parts = [_caltech101_labels(child) for child in ds.datasets]
            arr = np.concatenate(parts, axis=0)
        # 4) 최후 수단: __getitem__ 사용
        else:
            arr = np.array([int(ds[i][1]) for i in range(len(ds))], dtype=np.int64)
        
        return arr.astype(np.int64, copy=False)
    
    total_number_samples = len(dataset)
    labels = _caltech101_labels(dataset)
    num_classes = len(np.unique(labels))
    
    num_user_data = int(user_split * total_number_samples)
    
    if alpha is not None:
        method = 'dir'
    elif shard_per_user is not None:
        method = 'shard'
    else:
        method = 'dir'
        alpha = 0.5 if alpha is None else alpha
    
    print("caltech101_noniid", total_number_samples, num_classes, method)
    
    if method == "shard":
        dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
        idxs_dict = {}
        server_idx = np.random.choice(
            list(set(range(total_number_samples))),
            total_number_samples - num_user_data, 
            replace=False
        )
        local_idx = [i for i in range(total_number_samples) if i not in server_idx]
        
        for i in local_idx:
            y = int(labels[i])
            idxs_dict.setdefault(y, []).append(i)
        
        shard_per_class = int(shard_per_user * num_users / num_classes)
        for y in idxs_dict.keys():
            x = idxs_dict[y]
            num_leftover = len(x) % shard_per_class
            leftover = x[-num_leftover:] if num_leftover > 0 else []
            x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
            x = list(x.reshape((shard_per_class, -1)))
            
            for i, idx in enumerate(leftover):
                x[i] = np.concatenate([x[i], [idx]])
            idxs_dict[y] = x
        
        if proportions_dict is None:
            proportions_dict = list(range(num_classes)) * shard_per_class
            np.random.shuffle(proportions_dict)
            proportions_dict = np.array(proportions_dict).reshape((num_users, -1))
        
        for i in range(num_users):
            rand_set_label = proportions_dict[i]
            rand_set = []
            for y in rand_set_label:
                j = np.random.choice(len(idxs_dict[y]), replace=False)
                rand_set.append(idxs_dict[y].pop(j))
            dict_users[i] = np.concatenate(rand_set)
    
    elif method == "dir":
        if proportions_dict is None:
            proportions_dict = {k: None for k in range(num_classes)}
        
        # 서버와 로컬 인덱스 분할 (최소 보장 없이)
        server_idx = np.random.choice(
            list(range(total_number_samples)),
            total_number_samples - num_user_data, 
            replace=False
        )
        local_idx = np.array([
            i for i in range(total_number_samples)
            if i not in server_idx
        ])
        
        print(len(server_idx), len(local_idx))
        N = labels.shape[0]
        dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
        idx_batch = [[] for _ in range(num_users)]
        
        # 각 클래스별로 Dirichlet 분포를 사용해 분할
        for k in range(num_classes):
            idx_k_select = np.where(labels == k)[0]
            idx_k = [i for i in idx_k_select if i in local_idx]
            
            if len(idx_k) == 0:  # 해당 클래스에 로컬 데이터가 없으면 스킵
                continue
                
            np.random.shuffle(idx_k)
            
            if proportions_dict[k] is not None:
                proportions = proportions_dict[k]
            else:
                proportions = np.random.dirichlet(np.repeat(alpha, num_users))
                proportions_dict[k] = proportions
            
            # 균형 조정
            proportions = np.array([
                p * (len(x) < N / num_users) 
                for p, x in zip(proportions, idx_batch)
            ])
            proportions = proportions / proportions.sum()
            cuts = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            splits = np.split(idx_k, cuts)
            idx_batch = [x + s.tolist() for x, s in zip(idx_batch, splits)]
        
        # 최종 사용자별 데이터 할당
        for j in range(num_users):
            np.random.shuffle(idx_batch[j])
            dict_users[j] = np.array(idx_batch[j], dtype='int64')
    
    # 반환 형식을 다른 함수들과 동일하게 맞춤
    for i in range(num_users):
        dict_users[i] = list(dict_users[i])
    server_idx = list(server_idx)
    cnts_dict = {}
    client_losses = {}
    
    return dict_users, server_idx, cnts_dict, proportions_dict


class RemappedDataset(Dataset):
    """레이블 재매핑을 적용한 Dataset 래퍼"""
    
    def __init__(self, original_dataset, indices, label_mapping):
        self.original_dataset = original_dataset
        self.indices = indices
        self.label_mapping = label_mapping
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        img, label = self.original_dataset[original_idx]
        
        # 레이블 재매핑 적용
        remapped_label = self.label_mapping[label]
        
        return img, remapped_label