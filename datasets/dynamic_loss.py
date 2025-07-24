import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def compute_imbalance_ratio(labels, num_classes):
    """
    각 클라이언트의 데이터 분포를 분석하고, Long-Tail 여부를 판단하는 함수
    - labels: 해당 클라이언트의 데이터셋 (numpy array)
    - num_classes: 총 클래스 개수
    
    return: (imbalance_ratio, is_long_tail)
    """
    class_counts = np.array([np.count_nonzero(labels == j) for j in range(num_classes)])
    
    # 최소-최대 샘플 수를 기반으로 불균형 비율 계산 (Imbalance Ratio)
    max_count = np.max(class_counts)
    min_count = np.min(class_counts[class_counts > 0])  # 0을 제외한 최소값
    imbalance_ratio = max_count / min_count
    
    # 임계값을 기준으로 Long-Tail 여부 판단 (기본값: IR > 5이면 long-tail)
    IR_threshold = 5
    is_long_tail = imbalance_ratio > IR_threshold

    return imbalance_ratio, is_long_tail, class_counts

def gini_coefficient(class_counts):
    """
    Gini 계수를 사용하여 데이터 불균형 측정
    - class_counts: 각 클래스별 샘플 개수 (list 또는 numpy array)
    """
    class_counts = np.array(class_counts)
    sorted_counts = np.sort(class_counts)  # 샘플 개수를 정렬
    n = len(class_counts)
    cumulative = np.cumsum(sorted_counts)  # 누적 합
    gini = (2 * np.sum((np.arange(n) + 1) * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
    return gini



class DynamicLoss(nn.Module):
    def __init__(self, class_counts, gini_thresholds=(0.5, 0.7), gamma=1.0):
        """
        - class_counts: 각 클래스별 샘플 개수
        - ir_thresholds: (약한 long-tail 기준, 강한 long-tail 기준)
        - gamma: Focal Loss 감마 값
        """
        super(DynamicLoss, self).__init__()
        self.gini = gini_coefficient(class_counts)
        self.class_counts = class_counts
        self.gamma = gamma

        if self.gini <= gini_thresholds[0]:
            self.loss_fn = FocalLoss()
            self.loss_type = "FocalLoss"
        elif self.gini <= gini_thresholds[1]:
            self.loss_fn = FocalLoss()
            self.loss_type = "FocalLoss"
        else:
            self.loss_fn = FocalLoss()  # Severe Long-Tail에는 LDAM 적용
            self.loss_type = "FocalLoss"

        print(f"Dynamic Loss - Gini: {self.gini:.3f}, Loss Type: {self.loss_fn.__class__.__name__}")

    def forward(self, outputs, targets):
        return self.loss_fn(outputs, targets)
    
    
    

class ClassBalancedLoss(nn.Module):
    def __init__(self, class_counts, reduction='mean'):
        """
        - class_counts: 각 클래스별 샘플 개수 (numpy array 또는 list)
        - reduction: 'mean' 또는 'sum'
        """
        super(ClassBalancedLoss, self).__init__()
        self.class_weights = torch.tensor(1.0 / (class_counts + 1e-5), dtype=torch.float32)
        self.reduction = reduction

    def forward(self, outputs, targets):
        loss = F.cross_entropy(outputs, targets, weight=self.class_weights.to(outputs.device), reduction=self.reduction)
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification tasks with class imbalance.
    """
    def __init__(self, alpha=None, gamma=1.0, reduction='mean'):
        """
        :param alpha: Class-wise balancing factor (can be a list, tensor, or scalar).
        :param gamma: Focusing parameter to down-weight easy examples.
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha.float()
            else:
                self.alpha = torch.tensor([alpha], dtype=torch.float32)
        else:
            self.alpha = None
        
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Compute the focal loss.
        :param inputs: Predicted logits (not softmaxed) of shape (batch_size, num_classes).
        :param targets: Ground truth labels of shape (batch_size,).
        :return: Computed focal loss.
        """
        # Compute log-softmax
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        # Gather the log probabilities corresponding to the target classes
        target_log_probs = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
        target_probs = probs.gather(1, targets.view(-1, 1)).squeeze(1)
        
        # Compute the focal weight (1 - p_t)^gamma
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Compute the base cross entropy loss (negative log likelihood)
        loss = -target_log_probs * focal_weight
        
        # Apply class-wise alpha if given
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            loss *= alpha_t
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


        

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30):
        super().__init__()
        self.m_list = torch.tensor(1.0 / np.sqrt(np.sqrt(cls_num_list)), dtype=torch.float32)
        self.m_list = self.m_list * (max_m / torch.max(self.m_list))
        self.s = s

    def forward(self, outputs, targets):
        index = torch.zeros_like(outputs, dtype=torch.bool)
        index.scatter_(1, targets.data.view(-1, 1), 1)

        # 🔹 self.m_list를 outputs과 같은 디바이스로 이동
        m_list = self.m_list.to(outputs.device)[targets].view(-1, 1)

        outputs_m = outputs - index * m_list  # Margin 적용
        return F.cross_entropy(self.s * outputs_m, targets)

class Normalizer(): 
    def __init__(self, LpNorm=2, tau=1):
        self.LpNorm = LpNorm
        self.tau = tau
  
    def apply_on(self, model):  # tau-normalization을 classifier layer에 적용
        if hasattr(model, "fc"):  # fc 레이어가 있는지 확인
            curLayer = model.fc.weight
        elif hasattr(model, "classifier"):  # classifier 레이어가 있는 경우
            curLayer = model.classifier.weight
        else:
            raise AttributeError("Model does not have 'fc' or 'classifier' attribute.")

        curparam = curLayer.data
        curparam_vec = curparam.reshape((curparam.shape[0], -1))
        neuronNorm_curparam = (torch.linalg.norm(curparam_vec, ord=self.LpNorm, dim=1)**self.tau).detach().unsqueeze(-1)
        scalingVect = torch.ones_like(curparam)    

        idx = neuronNorm_curparam == neuronNorm_curparam
        idx = idx.squeeze()
        tmp = 1 / (neuronNorm_curparam[idx].squeeze())
        for _ in range(len(scalingVect.shape)-1):
            tmp = tmp.unsqueeze(-1)

        scalingVect[idx] = torch.mul(scalingVect[idx], tmp)
        curparam[idx] = scalingVect[idx] * curparam[idx]



class MDCSLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, tau=2):
        super().__init__()
        self.base_loss = F.cross_entropy

        prior = np.array(cls_num_list) #/ np.sum(cls_num_list)

        self.prior = torch.tensor(prior).float().cuda()
        self.C_number = len(cls_num_list)  # class number
        self.s = s
        self.tau = 2

        self.additional_diversity_factor = -0.2
        out_dim = 100
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center1", torch.zeros(1, out_dim))
        self.center_momentum = 0.9
        self.warmup = 20  
        self.reweight_epoch = 200
        if self.reweight_epoch != -1:
            idx = 1  # condition could be put in order to set idx
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float,
                                                        requires_grad=False)  # 这个是logits时算CE loss的weight
        self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float,
                                                              requires_grad=False).cuda()  # 这个是logits时算diversity loss的weight



    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)  # output_logits indicates the final prediction

        loss = 0
        temperature_mean = 1
        temperature = 1  
        # Obtain logits from each expert
        epoch = extra_info['epoch']
        num = int(target.shape[0] / 2)

        expert1_logits = extra_info['logits'][0] + torch.log(torch.pow(self.prior, -0.5) + 1e-9)      #head

        expert2_logits = extra_info['logits'][1] + torch.log(torch.pow(self.prior, 1) + 1e-9)         #medium

        expert3_logits = extra_info['logits'][2] + torch.log(torch.pow(self.prior, 2.5) + 1e-9)       #few



        teacher_expert1_logits = expert1_logits[:num, :]  # view1
        student_expert1_logits = expert1_logits[num:, :]  # view2

        teacher_expert2_logits = expert2_logits[:num, :]  # view1
        student_expert2_logits = expert2_logits[num:, :]  # view2

        teacher_expert3_logits = expert3_logits[:num, :]  # view1
        student_expert3_logits = expert3_logits[num:, :]  # view2




        teacher_expert1_softmax = F.softmax((teacher_expert1_logits) / temperature, dim=1).detach()
        student_expert1_softmax = F.log_softmax(student_expert1_logits / temperature, dim=1)

        teacher_expert2_softmax = F.softmax((teacher_expert2_logits) / temperature, dim=1).detach()
        student_expert2_softmax = F.log_softmax(student_expert2_logits / temperature, dim=1)

        teacher_expert3_softmax = F.softmax((teacher_expert3_logits) / temperature, dim=1).detach()
        student_expert3_softmax = F.log_softmax(student_expert3_logits / temperature, dim=1)


         

        teacher1_max, teacher1_index = torch.max(F.softmax((teacher_expert1_logits), dim=1).detach(), dim=1)
        student1_max, student1_index = torch.max(F.softmax((student_expert1_logits), dim=1).detach(), dim=1)

        teacher2_max, teacher2_index = torch.max(F.softmax((teacher_expert2_logits), dim=1).detach(), dim=1)
        student2_max, student2_index = torch.max(F.softmax((student_expert2_logits), dim=1).detach(), dim=1)

        teacher3_max, teacher3_index = torch.max(F.softmax((teacher_expert3_logits), dim=1).detach(), dim=1)
        student3_max, student3_index = torch.max(F.softmax((student_expert3_logits), dim=1).detach(), dim=1)


        # distillation
        partial_target = target[:num]
        kl_loss = 0
        if torch.sum((teacher1_index == partial_target)) > 0:
            kl_loss = kl_loss + F.kl_div(student_expert1_softmax[(teacher1_index == partial_target)],
                                         teacher_expert1_softmax[(teacher1_index == partial_target)],
                                         reduction='batchmean') * (temperature ** 2)

        if torch.sum((teacher2_index == partial_target)) > 0:
            kl_loss = kl_loss + F.kl_div(student_expert2_softmax[(teacher2_index == partial_target)],
                                         teacher_expert2_softmax[(teacher2_index == partial_target)],
                                         reduction='batchmean') * (temperature ** 2)

        if torch.sum((teacher3_index == partial_target)) > 0:
            kl_loss = kl_loss + F.kl_div(student_expert3_softmax[(teacher3_index == partial_target)],
                                         teacher_expert3_softmax[(teacher3_index == partial_target)],
                                         reduction='batchmean') * (temperature ** 2)

        loss = loss + 0.6 * kl_loss * min(extra_info['epoch'] / self.warmup, 1.0)



        # expert 1
        loss += self.base_loss(expert1_logits, target)

        # expert 2
        loss += self.base_loss(expert2_logits, target)

        # expert 3
        loss += self.base_loss(expert3_logits, target)


        return loss

    @torch.no_grad()
    def update_center(self, center, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output))  # * dist.get_world_size())

        # ema update

        return center * self.center_momentum + batch_center * (1 - self.center_momentum)
    
    
class VSLoss(nn.Module):

    def __init__(self, cls_num_list, gamma=0.3, tau=1.0, weight=None):
        super(VSLoss, self).__init__()

        cls_probs = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]
        temp = (1.0 / np.array(cls_num_list)) ** gamma
        temp = temp / np.min(temp)

        iota_list = tau * np.log(cls_probs)
        Delta_list = temp

        self.iota_list = torch.cuda.FloatTensor(iota_list)
        self.Delta_list = torch.cuda.FloatTensor(Delta_list)
        self.weight = weight

    def forward(self, x, target, use_multiplicative=True):
        output = x / self.Delta_list + self.iota_list if use_multiplicative else x + self.iota_list

        return F.cross_entropy(output, target, weight=self.weight)