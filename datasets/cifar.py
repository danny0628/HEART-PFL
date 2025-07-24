# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/PerAda/blob/main/LICENSE

import os 
from PIL import Image
import numpy as np 
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import random

class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root :str, name :str,
                 transform=None, target_transform=None):
        super(CIFAR10C, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')
        
        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        self.targets = self.targets.astype(int)
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img, targets
    
    def __len__(self):
        return len(self.data)



class CIFAR101(datasets.VisionDataset):
    def __init__(self, root :str,
                 transform=None, target_transform=None):
        super(CIFAR101, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )

       
        data_path = os.path.join(root, 'cifar10.1_v6_data.npy')
        target_path = os.path.join(root,  'cifar10.1_v6_labels.npy')

        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        self.targets = self.targets.astype(int)
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img, targets
    
    def __len__(self):
        return len(self.data)

class CIFAR100LT(Dataset):
    def __init__(self, root, split="train", transform=None, target_transform=None, imb_factor=0.01):
        """
        CIFAR-100-LT 데이터셋 생성

        :param root: 데이터 경로
        :param split: "train" 또는 "test"
        :param transform: 이미지 변환
        :param target_transform: 라벨 변환
        :param imb_factor: 불균형 비율 (작을수록 long-tailed)
        """
        # CIFAR-100 불러오기
        self.dataset = datasets.CIFAR100(root=root, train=(split == "train"),
                                         download=True, transform=transform, target_transform=target_transform)
        self.targets = np.array(self.dataset.targets)

        # Long-Tailed 분포 생성
        self._create_long_tail(imb_factor)

    def _create_long_tail(self, imb_factor):
        num_classes = 100
        num_per_class = 500  # CIFAR-100의 각 클래스 당 샘플 수 (Train 기준)

        # Long-Tailed 분포 설정
        class_counts = [int(num_per_class * (imb_factor ** (i / (num_classes - 1)))) for i in range(num_classes)]

        # 샘플 선택
        new_data = []
        new_targets = []
        for i in range(num_classes):
            idx = np.where(self.targets == i)[0]
            selected_idx = np.random.choice(idx, class_counts[i], replace=False)
            new_data.extend([self.dataset.data[j] for j in selected_idx])
            new_targets.extend([i] * len(selected_idx))

        # 업데이트된 데이터셋 적용
        self.dataset.data = np.array(new_data)
        self.dataset.targets = np.array(new_targets)

    def __getitem__(self, index):
        img, target = self.dataset.data[index], self.dataset.targets[index]
        return self.dataset.transform(img), target

    def __len__(self):
        return len(self.dataset.targets)
