# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file: https://github.com/facebookresearch/FL_partial_personalization/blob/main/LICENSE

import torch
import torch.nn as nn
from typing import Optional
from torchinfo import summary
import torch.nn.functional as F
from .base_model import PFLBaseModel


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

GROUP_NORM_LOOKUP = {
    16: 2,  # -> channels per group: 8
    32: 4,  # -> channels per group: 8
    64: 8,  # -> channels per group: 8
    128: 8,  # -> channels per group: 16
    256: 16,  # -> channels per group: 16
    512: 32,  # -> channels per group: 16
    1024: 32,  # -> channels per group: 32
    2048: 32,  # -> channels per group: 64
}

class BatchNorm2d(nn.BatchNorm2d):
    """Official NBN BatchNorm2d - scale 메서드 포함"""
    def __init__(self, dim):
        super().__init__(dim)
        
    def scale(self, alpha):
        self.w = self.weight / (torch.norm(self.weight)+1e-6) * alpha
        if self.bias is not None:
            self.b = self.bias / (torch.norm(self.bias)+1e-6) * alpha
        else:
            self.b = self.bias
            
    def forward(self, input):
        self._check_input_dim(input)

        w = self.w if hasattr(self, 'w') else self.weight
        b = self.b if hasattr(self, 'b') else self.bias
        
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        x = F.batch_norm(
            input,
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            w,
            b,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        return x

def create_batch_norm(num_channels):
    # return torch.nn.GroupNorm(GROUP_NORM_LOOKUP[num_channels], num_channels)
    return torch.nn.BatchNorm2d(num_channels)

class ResidualBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None) -> None:
        super(ResidualBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.planes = planes
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = create_batch_norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = create_batch_norm(planes)
        self.downsample = downsample
        self.stride = stride

        self.use_adapter = False
        self.adapter1 = False
        self.adapter2 = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        if self.use_adapter:
            out = self.adapter1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.use_adapter:
            out = self.adapter2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

    def add_adapters(self, dropout=0.):
        if not self.use_adapter:
            self.use_adapter = True
            self.adapter1 = AdapterBlockNBN(self.planes, dropout)
            self.adapter2 = AdapterBlockNBN(self.planes, dropout)

class AdapterBlock(nn.Module):
    def __init__(self, planes, dropout):
        super().__init__()
        # self.bn = nn.BatchNorm2d(planes)
        self.bn = create_batch_norm(planes) ## 이게 default임!!!!!!!!!!!!!!
        # self.bn = BatchNorm2d(planes) ### 이게 NBN 적용하는 버전
        self.conv = conv1x1(planes, planes)  # 1x1 convolution / LoRA 유무에따라 들어감 
        # self.lora = LoRAConv2d(planes,planes,r=4, alpha=1.0)
        self.dropout = nn.Dropout(dropout)
        # initialize
        nn.init.normal_(self.conv.weight, 0, 1e-4) # LoRA 유무에따라 들어감 
        # nn.init.normal_(self.lora.A.weight, mean=0, std=1e-4)
        # nn.init.zeros_(self.lora.B.weight)
        # nn.init.constant_(self.conv.bias, 0.0)  # no bias

    def forward(self, x):
        identity = x
        out = self.bn(x)  # Batch norm
        out = self.conv(self.dropout(out))  # 1x1 conv / LoRA 유무에따라 들어감 
        # out = self.dropout(out)
        # out = self.lora(out)
        out += identity  # skip connection
        return out

class AdapterBlockNBN(nn.Module):
    """NBN이 적용된 Adapter Block - Official 스타일"""
    def __init__(self, planes, dropout):
        super().__init__()
        # Official NBN BatchNorm 사용
        self.bn = BatchNorm2d(planes)
        self.conv = conv1x1(planes, planes)
        self.dropout = nn.Dropout(dropout)
        
        # 초기화
        nn.init.normal_(self.conv.weight, 0, 1e-4)

    def forward(self, x):
        identity = x
        out = self.bn(x)  # NBN이 적용된 Batch norm
        out = self.conv(self.dropout(out))
        out += identity
        return out

    
class ResNetBN(PFLBaseModel):
    def __init__(self, layers=(2, 2, 2, 2), num_classes=62, original_size=False,save_activations=False ):
        # if original_size: expect (3, 224, 224) images, else expect (1, 28, 28)
        super().__init__()
        self.inplanes = 64
        self.drop_i = nn.Dropout(0.)
        self.drop_o = nn.Dropout(0.)
        if original_size:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = torch.nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = create_batch_norm(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        if original_size:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.Identity()
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.is_on_client = None
        self.is_on_server = None
        self.save_activations= save_activations
        
        self.activations = None

    def _make_layer(self, planes: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                # nn.BatchNorm2d(planes),
                create_batch_norm(planes),
            )
        layers = []
        layers.append(ResidualBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def init_scale_factor(self):
        """Adapter 추가 후 NBN scale factor 초기화"""
        module_list = []
        for n, m in self.named_modules():
            # adapter 내부의 BatchNorm2d만 찾기
            if hasattr(m, "scale") and isinstance(m, BatchNorm2d) and 'adapter' in n:
                module_list.append(n)
        
        if module_list:
            self.scale_factor = nn.Parameter(torch.ones((len(module_list),)))
            self.module_list = module_list
            print(f"NBN initialized for {len(module_list)} adapter BatchNorm modules")
    
    def scale_modules(self):
        """Forward pass 전에 adapter의 BatchNorm에만 scale 적용"""
        if self.scale_factor is not None:
            idx = 0
            for n, m in self.named_modules():
                if n in self.module_list:
                    m.scale(self.scale_factor[idx])
                    idx += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.scale_modules()
        
        x = self.conv1(self.drop_i(x))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        rb1 = self.layer1(x)
        rb2 = self.layer2(rb1)
        rb3 = self.layer3(rb2)
        rb4 = self.layer4(rb3)

        x = self.avgpool(rb4)
        x = torch.flatten(x, 1)
        x = self.fc(self.drop_o(x))
        if self.save_activations:
            self.activations= [rb1,rb2,rb3,rb4]
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(self.drop_i(x))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # 🔹 Feature Extraction: 마지막 Conv Layer의 출력 반환

        x = self.avgpool(x)  # 🔹 평균 풀링 적용
        x = torch.flatten(x, 1)  # 🔹 벡터화
        return x  # 🔹 Feature Map 반환
    
    def extract_features_all(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(self.drop_i(x))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        
        pooled_feats = []
        for f in [f1, f2, f3, f4]:
            pooled = self.avgpool(f)
            pooled = torch.flatten(pooled, 1)
            pooled_feats.append(pooled)
        
        return pooled_feats

    def print_summary(self, train_batch_size):
        raise NotImplementedError

    def split_server_and_client_params(self, client_mode, layers_to_client, adapter_hidden_dim=-1, dropout=0.):
        device = next(self.parameters()).device
        if self.is_on_client is not None:
            raise ValueError('This model has already been split across clients and server.')
        assert client_mode in ['none', 'res_layer', 'inp_layer', 'out_layer', 'adapter', 'interpolate', 'finetune'] 
        # Prepare
        if layers_to_client is None:  # no layers to client
            layers_to_client = []
        if client_mode == 'res_layer' and len(layers_to_client) is None:
            raise ValueError(f'No residual blocks to finetune. Nothing to do')
        is_on_server = None
        
        # Set requires_grad based on `train_mode`
        if client_mode in ['none', None]:
            # no parameters on the client
            def is_on_client(name):
                return False
        elif 'res_layer' in client_mode:
            # Specific residual blocks are sent to client (available layers are [1, 2, 3, 4])
            def is_on_client(name):
                return any([f'layer{i}' in name for i in layers_to_client])
        elif client_mode in ['inp_layer']:
            # First convolutional layer is sent to client
            def is_on_client(name):
                return (name in ['conv1.weight', 'bn1.weight', 'bn1.bias'])  # first conv + bn
            self.drop_i = nn.Dropout(dropout)
        elif client_mode in ['out_layer']:
            # Final linear layer is sent to client
            def is_on_client(name):
                return (name in ['fc.weight', 'fc.bias'])  # final fc
            self.drop_o = nn.Dropout(dropout)
        # elif client_mode in ['adapter']:
        #     # Train adapter modules (+ batch norm)
        #     def is_on_client(name):
        #         return ('adapter' in name) or ('bn1' in name) or ('bn2' in name)
        #     # Add adapter modules
        #     for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
        #         for block in layer.children():
        #             # each block is of type `ResidualBlock`
        #             block.add_adapters(dropout)
        if client_mode in ['adapter']:
            # Train adapter modules (+ batch norm + scale factor)
            def is_on_client(name):
                return ('adapter' in name) or ('scale_factor' in name)
            
            # Add adapter modules with NBN
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for block in layer.children():
                    block.add_adapters(dropout)
            
            # Adapter 추가 후 NBN scale factor 초기화
            self.init_scale_factor()        
        
        elif client_mode == 'interpolate':  # both on client and server
            is_on_client = lambda _: True
            is_on_server = lambda _: True
        elif client_mode == 'finetune':  # all on client
            is_on_client = lambda _: True
            is_on_server = lambda _: False
        else:
            raise ValueError(f'Unknown client_mode: {client_mode}')
        if is_on_server is None:
            def is_on_server(name): 
                return not is_on_client(name)
        
        self.is_on_client = is_on_client
        self.is_on_server = is_on_server
        self.to(device)
    

