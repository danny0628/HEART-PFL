# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/PerAda/blob/main/LICENSE

# Part of the implementation is based on: https://github.com/epfml/federated-learning-public-code/blob/master/codes/FedDF-code/pcode/aggregation/noise_knowledge_transfer.py

import copy 
import torch 
from utils.stat_tracker import BestPerf 
import collections
import torch.nn.functional as F
import torchattacks
import torch.nn as nn


cifar100_to_cifar10_labels = {
    0: 4, 1: 1, 2: 7, 3: 7, 4: 6, 5: 7, 6: 6, 7: 5, 8: 9, 9: 9,
    10: 4, 11: 2, 12: 0, 13: 9, 14: 1, 15: 5, 16: 3, 17: 8, 18: 5, 19: 8,
    20: 6, 21: 0, 22: 2, 23: 3, 24: 3, 25: 4, 26: 5, 27: 6, 28: 2, 29: 0,
    30: 1, 31: 1, 32: 7, 33: 7, 34: 3, 35: 8, 36: 4, 37: 2, 38: 9, 39: 9,
    40: 3, 41: 8, 42: 8, 43: 3, 44: 4, 45: 2, 46: 6, 47: 2, 48: 6, 49: 5,
    50: 4, 51: 4, 52: 0, 53: 9, 54: 1, 55: 5, 56: 7, 57: 2, 58: 0, 59: 8,
    60: 8, 61: 6, 62: 5, 63: 6, 64: 2, 65: 3, 66: 3, 67: 1, 68: 1, 69: 9,
    70: 7, 71: 8, 72: 3, 73: 0, 74: 3, 75: 6, 76: 7, 77: 5, 78: 0, 79: 7,
    80: 3, 81: 1, 82: 4, 83: 8, 84: 5, 85: 0, 86: 9, 87: 8, 88: 4, 89: 2,
    90: 2, 91: 6, 92: 1, 93: 9, 94: 2, 95: 0, 96: 7, 97: 9, 98: 4, 99: 5
}

filtered_cifar100_to_cifar10_labels = {
    48: 0,   # airplane → airplane
    14: 1,   # bus → automobile
    51: 1,   # motorcycle → automobile
    8:  1,   # bicycle → automobile
    58: 9,   # pickup truck → truck
    73: 2,   # sparrow → bird
    88: 2,   # rooster → bird
    94: 2,   # woodpecker → bird
    62: 2,   # parrot → bird
    82: 2,   # pigeon → bird
    66: 3,   # kitten → cat
    31: 4,   # deer → deer
    95: 5,   # wolf → dog
    70: 5,   # fox → dog
    36: 6,   # frog → frog
    46: 7,   # horse → horse
    64: 8,   # ship → ship
    75: 9    # tractor → truck
}


def computeAUROC(dataGT, dataPRED, nnClassCount):
    # Computes area under ROC curve 
    # dataGT: ground truth data
    # dataPRED: predicted data
    outAUROC = []
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
    
    from sklearn.metrics import roc_auc_score

    for i in range(nnClassCount):
        try:
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
        except ValueError:
            pass
    return outAUROC


def test(dataset, model, testloader , log=False):
    model.eval()

    test_loss = 0
    correct = 0
    total = 0
    if dataset =='chexpert':
        y_prob = []
        y_true = []
        from sklearn import metrics
        import numpy as np 
        criterion= torch.nn.BCEWithLogitsLoss()
        sgmd =torch.nn.Sigmoid().cuda()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(testloader):
                x, y = x.to('cuda'), y.to('cuda')
                pred = model(x)
                loss = criterion(pred, y)
                y_prob.append(sgmd(pred).detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())

                test_loss += loss.item()
                total += y.size(0)
                
            y_prob = np.concatenate(y_prob, axis=0)
            y_true = np.concatenate(y_true, axis=0)
            aurocMean = metrics.roc_auc_score(y_true, y_prob, average='macro')
            if log:
                print('test Loss: %.3f | Acc:%.3f%%'%(test_loss/(batch_idx+1), aurocMean*100))
        return test_loss/len(testloader), aurocMean*100.

    else:
        criterion= torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(testloader):
                x, y = x.to('cuda'), y.to('cuda')
                pred = model(x)
                loss = criterion(pred, y)

                test_loss += loss.item()
                _, pred_c = pred.max(1)
                total += y.size(0)
                correct += pred_c.eq(y).sum().item()
            if log:
                print('test Loss: %.3f | Acc:%.3f%%'%(test_loss/(batch_idx+1), 100.*correct/total))

        acc = 100.*correct/total
        return test_loss/len(testloader), acc



def divergence(student_logits, teacher, kd_KL_temperature=1, use_teacher_logits=True):
    divergence = F.kl_div(
        F.log_softmax(student_logits / kd_KL_temperature, dim=1),
        F.softmax(teacher/ kd_KL_temperature, dim=1)
        if use_teacher_logits
        else teacher,
        reduction="batchmean",
    )  # forward KL
    return divergence

def divergence_reverse(student_logits, teacher, kd_KL_temperature=1, use_teacher_logits=True):
    divergence = F.kl_div(
        F.softmax(teacher/ kd_KL_temperature, dim=1),
        F.log_softmax(student_logits / kd_KL_temperature, dim=1)
        if use_teacher_logits
        else teacher,
        reduction="batchmean",
    )  # forward KL
    return divergence

def check_early_stopping(
        model,
        model_ind,
        best_tracker,
        validated_perf,
        validated_perfs,
        perf_index,
        early_stopping_batches,
        log_fn=print,
        best_models=None,
    ):
    # update the tracker.
    best_tracker.update(perf=validated_perf, perf_location=perf_index)
    if validated_perfs is not None:
        validated_perfs[model_ind].append(validated_perf)

    # save the best model.
    if best_tracker.is_best and best_models is not None:
        best_models[model_ind] = copy.deepcopy(model)

    # check if we need the early stopping or not.
    if perf_index - best_tracker.get_best_perf_loc >= early_stopping_batches:
        log_fn(
            f"\tMeet the early stopping condition (batches={early_stopping_batches}): early stop!! (perf_index={perf_index}, best_perf_loc={best_tracker.get_best_perf_loc})."
        )
        return True
    else:
        return False


def Distillation_mapping(args, kd_server_smodel, kd_trainset, val_dataloader, local_model_list):
    import collections
    import torchattacks
    import copy
    import torch.nn.functional as F

    # ✅ CIFAR-100의 유사 클래스만 필터링한 매핑
    filtered_cifar100_to_cifar10_labels = {
        48: 0, 58: 9, 14: 1, 51: 1, 8: 1, 88: 2, 62: 2, 73: 2, 82: 2, 94: 2,
        66: 3, 31: 4, 95: 5, 70: 5, 36: 6, 46: 7, 64: 8, 75: 9
    }

    kd_lr = args.kd_lr
    kd_eval_batches_freq = args.kd_eval_batches_freq
    early_stopping_server_batches = args.early_stopping_server_batches
    total_n_server_pseudo_batches = args.total_n_server_pseudo_batches
    kd_KL_temperature = args.kd_KL_temperature

    optimizer_server_student = torch.optim.Adam(
        filter(lambda p: p.requires_grad, kd_server_smodel.parameters()), lr=kd_lr)
    scheduler_server_student = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_server_student, total_n_server_pseudo_batches, last_epoch=-1)

    kd_model = kd_server_smodel
    kd_server_smodel.eval()
    _, init_perf_on_val = test(args.dataset, kd_server_smodel, val_dataloader)
    print("init_perf_on_val: ", init_perf_on_val)

    server_best_tracker = BestPerf(best_perf=None, larger_is_better=True)
    best_models = [None]
    validated_perfs = collections.defaultdict(list)
    distillation_data_loader = kd_trainset

    print("use kd dataset", args.kd_dataset)
    data_iter = iter(distillation_data_loader)
    batch_id = 0

    al_global = copy.deepcopy(kd_model)
    al_global.train()
    adversary_global = torchattacks.PGD(al_global, steps=20)
    local_model_list = [model.cuda() for model in local_model_list]
    adversary_list = [torchattacks.PGD(model, steps=20) for model in local_model_list]

    while batch_id < total_n_server_pseudo_batches:
        try:
            batch_data = next(data_iter)
        except StopIteration:
            data_iter = iter(distillation_data_loader)
            batch_data = next(data_iter)

        raw_inputs = batch_data[0]
        raw_labels = batch_data[1].cpu().numpy()

        if args.kd_dataset == "cifar100":
            valid_indices = [i for i, lbl in enumerate(raw_labels)
                             if lbl in filtered_cifar100_to_cifar10_labels]
            if len(valid_indices) == 0:
                continue

            filtered_inputs = raw_inputs[valid_indices].cuda()
            filtered_labels = [filtered_cifar100_to_cifar10_labels[raw_labels[i]] for i in valid_indices]
            mapped_labels = torch.tensor(filtered_labels).cuda()

            pseudo_data_student = filtered_inputs
            pseudo_data_teacher = filtered_inputs
        else:
            pseudo_data_student = batch_data[0].cuda()
            pseudo_data_teacher = pseudo_data_student
            mapped_labels = batch_data[1].cuda()

        out_t = None
        out_t_al = None

        for cli_idx in range(len(local_model_list)):
            al_net = local_model_list[cli_idx]
            adversary = adversary_list[cli_idx]

            al_batch_data = adversary(pseudo_data_teacher, mapped_labels)
            pseudo_data_student_al = adversary_global(pseudo_data_student, mapped_labels)
            pseudo_data_teacher_al = al_batch_data.cuda()

            with torch.no_grad():
                _logits = al_net(pseudo_data_teacher)
                _logits_al = al_net(pseudo_data_teacher_al)

                if out_t is not None:
                    out_t += _logits / len(local_model_list)
                else:
                    out_t = _logits / len(local_model_list)

                if out_t_al is not None:
                    out_t_al += _logits_al / len(local_model_list)
                else:
                    out_t_al = _logits_al / len(local_model_list)

        kd_server_smodel.train()
        optimizer_server_student.zero_grad()

        out_s = kd_server_smodel(pseudo_data_student)
        out_s_al = kd_server_smodel(pseudo_data_student_al)

        loss_clean = F.kl_div(
            F.log_softmax(out_s / kd_KL_temperature, dim=1),
            F.softmax(out_t / kd_KL_temperature, dim=1),
            reduction="batchmean",
        )
        loss_al = F.kl_div(
            F.log_softmax(out_s_al / kd_KL_temperature, dim=1),
            F.softmax(out_t_al / kd_KL_temperature, dim=1),
            reduction="batchmean",
        )
        loss_clean_re = F.kl_div(
            F.softmax(out_t / kd_KL_temperature, dim=1),
            F.log_softmax(out_s / kd_KL_temperature, dim=1),
            reduction="batchmean",
        )
        loss_al_re = F.kl_div(
            F.softmax(out_t_al / kd_KL_temperature, dim=1),
            F.log_softmax(out_s_al / kd_KL_temperature, dim=1),
            reduction="batchmean",
        )

        loss = loss_clean + loss_al + loss_clean_re + loss_al_re
        loss.backward()
        optimizer_server_student.step()
        scheduler_server_student.step()

        if (batch_id + 1) % kd_eval_batches_freq == 0:
            kd_server_smodel.eval()
            _, validated_perf = test(args.dataset, kd_server_smodel, val_dataloader, log=False)
            print(f'Server Batch[{batch_id + 1}/{total_n_server_pseudo_batches}] '
                  f'KD Loss: {loss:.4f}, ValAcc: {validated_perf:.2f}')

            if check_early_stopping(
                model=kd_server_smodel,
                model_ind=0,
                best_tracker=server_best_tracker,
                validated_perf=validated_perf,
                validated_perfs=validated_perfs,
                perf_index=batch_id + 1,
                early_stopping_batches=early_stopping_server_batches,
                best_models=best_models,
            ):
                break

        batch_id += 1

    if init_perf_on_val >= server_best_tracker.best_perf:
        print("use init server model instead.")
        return None, 0
    else:
        print(f"use distillation model at server step {server_best_tracker.get_best_perf_loc} "
              f"with val performance {server_best_tracker.best_perf:.2f}")
        kd_server_smodel = best_models[0]
        return kd_server_smodel, server_best_tracker.get_best_perf_loc - 1


        
def Distillation(args, kd_server_smodel, kd_trainset , val_dataloader, local_model_list):
    kd_lr = args.kd_lr
    kd_eval_batches_freq= args.kd_eval_batches_freq 
    
    early_stopping_server_batches =args.early_stopping_server_batches
    total_n_server_pseudo_batches =args.total_n_server_pseudo_batches
    kd_KL_temperature =args.kd_KL_temperature

    
    optimizer_server_student = torch.optim.Adam(filter(lambda p: p.requires_grad, kd_server_smodel.parameters()),lr = kd_lr)
    scheduler_server_student = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_server_student,
        total_n_server_pseudo_batches,
        last_epoch=-1,
    )
    
    kd_model = kd_server_smodel
    # get the init server perf.
    kd_server_smodel.eval()
    _, init_perf_on_val = test(args.dataset, kd_server_smodel,val_dataloader )
    print("init_perf_on_val: ", init_perf_on_val)

    server_best_tracker = BestPerf(best_perf=None, larger_is_better=True)
    best_models = [None]
    validated_perfs = collections.defaultdict(list)
    ## note that kd_trainset is replaced to  kd_dataloader 
    distillation_data_loader=kd_trainset 
   
    print("use kd dataset", args.kd_dataset)   

    data_iter = iter(distillation_data_loader)
    batch_id= 0


    al_global = copy.deepcopy(kd_model)
    al_global.train()
    adversary_global = torchattacks.PGD(al_global, steps=20)
    local_model_list = [model.cuda() for model in local_model_list]
    adversary_list = [torchattacks.PGD(model, steps=20) for model in local_model_list]

    while batch_id < total_n_server_pseudo_batches:
        try:
            batch_data = next(data_iter)
        except StopIteration:
            data_iter = iter(distillation_data_loader)
            batch_data = next(data_iter)
        
        
        pseudo_data_student=batch_data[0].cuda() # 0 is data, 1 is label
        pseudo_data_teacher = pseudo_data_student
        
        
        out_t= None
        out_t_al = None
        
        ## cifar100 --> cifar10 match과정
        # if args.kd_dataset == "cifar100":
        #     batch_labels = batch_data[1].cpu().numpy()
        #     mapped_labels = torch.tensor([cifar100_to_cifar10_labels[label] for label in batch_labels]).cuda()
        
        # al_global = copy.deepcopy(kd_model)
        
        
        for cli_idx in range(len(local_model_list)):
            
            ## adv
            al_net = local_model_list[cli_idx]
            adversary = adversary_list[cli_idx]
            
            ## Feature map KD loss 실험
            # feature_t = local_model_list[cli_idx].extract_features(pseudo_data_teacher)
            # feature_s = kd_server_smodel.extract_features(pseudo_data_student)
            # layer_diffs = [torch.norm(F_T - F_S) for F_T, F_S in zip(feature_t, feature_s)]
            # total_diff = sum(layer_diffs)
            # layer_weights = [diff / (total_diff + 1e-8) for diff in layer_diffs]  # Normalize weights

            # Feature Distillation Loss 계산
            # loss_feat = sum(w * torch.nn.functional.mse_loss(F_T, F_S) for w, F_T, F_S in zip(layer_weights, feature_t, feature_s))    
                        
            
            # if args.kd_dataset == "cifar10":
            al_batch_data = adversary(batch_data[0].cuda(), batch_data[1].cuda())
            pseudo_data_student_al = adversary_global(batch_data[0].cuda(), batch_data[1].cuda())
            # elif args.kd_dataset == "cifar100":
            #     al_batch_data = adversary(batch_data[0].cuda(), mapped_labels)
            #     pseudo_data_student_al = adversary_global(batch_data[0].cuda(), mapped_labels)
            # elif args.kd_dataset == "inat2018":
            #     al_batch_data = adversary(batch_data[0].cuda(), mapped_labels)
            #     pseudo_data_student_al = adversary_global(batch_data[0].cuda(), mapped_labels)
                
            pseudo_data_teacher_al = al_batch_data.cuda()
           

            with torch.no_grad():
                _logits =  local_model_list[cli_idx](pseudo_data_teacher)
                _logits_al = al_net(pseudo_data_teacher_al)
                # KL loss
                if out_t is not None:
                    out_t += _logits * 1/ len(local_model_list)
                else: 
                    out_t = _logits * 1/ len(local_model_list)
                    
                if out_t_al is not None:
                    out_t_al += _logits_al * 1/ len(local_model_list)
                else:
                    out_t_al = _logits_al * 1/len(local_model_list)
            
                
        kd_server_smodel.train()
        # steps on the same pseudo data
        optimizer_server_student.zero_grad()
        out_s =  kd_server_smodel(pseudo_data_student)     
        out_s_al = kd_server_smodel(pseudo_data_student_al)
        # KL loss
        loss_clean = divergence( #   Distilling the Knowledge in a Neural Network
                out_s, out_t, kd_KL_temperature
            )
        loss_al = divergence(
            out_s_al, out_t_al, kd_KL_temperature
        )
        loss_clean_re = divergence_reverse(out_s, out_t, kd_KL_temperature)
        loss_al_re = divergence_reverse(out_s_al, out_t_al, kd_KL_temperature)
        
        loss = loss_clean + loss_al + loss_clean_re + loss_al_re
        # loss = loss_clean + loss_al
    
        loss.backward()
      
        optimizer_server_student.step()

        # after each batch.
        if scheduler_server_student is not None:
            scheduler_server_student.step()
        # overfit need early stop
        
        if (batch_id+1) % kd_eval_batches_freq == 0:
            kd_server_smodel.eval()
            _, validated_perf = test(args.dataset,
                 kd_server_smodel, val_dataloader,log=False
            )
            log_str = ('Server Batch[{0:03}/{1:03}] '
                    'KD:{kd_loss:.4f} ValAcc{val_acc}'.format(
                    batch_id, total_n_server_pseudo_batches ,kd_loss= loss, val_acc =validated_perf   ))
            print(log_str)

              # check early stopping.
            if check_early_stopping(
                model=kd_server_smodel,
                model_ind=0,
                best_tracker=server_best_tracker,
                validated_perf=validated_perf,
                validated_perfs=validated_perfs,
                perf_index=batch_id + 1,
                early_stopping_batches=early_stopping_server_batches,
                best_models=best_models,
            ):
                break  

        batch_id += 1

    use_init_server_model = (
            True
            if init_perf_on_val  >= server_best_tracker.best_perf
            else False
        )

        # get the server model.
    if use_init_server_model:
        print("use init server model instead.")
        return None, 0
    else:
        print("use distillation model at server step {} with val performance {}".format(server_best_tracker.get_best_perf_loc,server_best_tracker.best_perf  ))
        kd_server_smodel = best_models[0] 
        return kd_server_smodel, server_best_tracker.get_best_perf_loc -1

    
    
    
def Distillation_origin(args, kd_server_smodel, kd_trainset , val_dataloader, local_model_list):
    kd_lr = args.kd_lr
    kd_eval_batches_freq= args.kd_eval_batches_freq 
    
    early_stopping_server_batches =args.early_stopping_server_batches
    total_n_server_pseudo_batches =args.total_n_server_pseudo_batches
    kd_KL_temperature =args.kd_KL_temperature

    
    optimizer_server_student = torch.optim.Adam(filter(lambda p: p.requires_grad, kd_server_smodel.parameters()),lr = kd_lr)
    scheduler_server_student = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_server_student,
        total_n_server_pseudo_batches,
        last_epoch=-1,
    )
    
    
    # get the init server perf.
    kd_server_smodel.eval()
    _, init_perf_on_val = test(args.dataset, kd_server_smodel,val_dataloader )
    print("init_perf_on_val: ", init_perf_on_val)

    server_best_tracker = BestPerf(best_perf=None, larger_is_better=True)
    best_models = [None]
    validated_perfs = collections.defaultdict(list)
    ## note that kd_trainset is replaced to  kd_dataloader 
    distillation_data_loader=kd_trainset 
   
    print("use kd dataset", args.kd_dataset)   

    data_iter = iter(distillation_data_loader)
    batch_id= 0



    while batch_id < total_n_server_pseudo_batches:
        try:
            batch_data = next(data_iter)
        except StopIteration:
            data_iter = iter(distillation_data_loader)
            batch_data = next(data_iter)
        

        pseudo_data_student=batch_data[0].cuda() # 0 is data, 1 is label
        pseudo_data_teacher = pseudo_data_student


        out_t= None
        
        for cli_idx in range(len(local_model_list)):     
            with torch.no_grad():
                _logits =  local_model_list[cli_idx](pseudo_data_teacher)
                
                # KL loss
                if out_t is not None:
                    out_t += _logits * 1/ len(local_model_list)
                else: 
                    out_t = _logits * 1/ len(local_model_list)
                


        kd_server_smodel.train()
        # steps on the same pseudo data
        optimizer_server_student.zero_grad()
        out_s =  kd_server_smodel(pseudo_data_student)     
   
        # KL loss
        loss = divergence( #   Distilling the Knowledge in a Neural Network
                out_s, out_t, kd_KL_temperature
            )
    
        loss.backward()
      
        optimizer_server_student.step()

        # after each batch.
        if scheduler_server_student is not None:
            scheduler_server_student.step()
        # overfit need early stop
        
        if (batch_id+1) % kd_eval_batches_freq == 0:
            kd_server_smodel.eval()
            _, validated_perf = test(args.dataset,
                 kd_server_smodel, val_dataloader,log=False
            )
            log_str = ('Server Batch[{0:03}/{1:03}] '
                    'KD:{kd_loss:.4f} ValAcc{val_acc}'.format(
                    batch_id, total_n_server_pseudo_batches ,kd_loss= loss, val_acc =validated_perf   ))
            print(log_str)

              # check early stopping.
            if check_early_stopping(
                model=kd_server_smodel,
                model_ind=0,
                best_tracker=server_best_tracker,
                validated_perf=validated_perf,
                validated_perfs=validated_perfs,
                perf_index=batch_id + 1,
                early_stopping_batches=early_stopping_server_batches,
                best_models=best_models,
            ):
                break

        batch_id += 1

    use_init_server_model = (
            True
            if init_perf_on_val  >= server_best_tracker.best_perf
            else False
        )

        # get the server model.
    if use_init_server_model:
        print("use init server model instead.")
        return None, 0
    else:
        print("use distillation model at server step {} with val performance {}".format(server_best_tracker.get_best_perf_loc,server_best_tracker.best_perf  ))
        kd_server_smodel = best_models[0] 
        return kd_server_smodel, server_best_tracker.get_best_perf_loc -1 
    
import sys
sys.path.append('/mnt/home/mjkim1/node7.gpu/miccai2025_fl/aggregation')
from AT_helper import adaad_inner_loss

def DGAD(args, kd_server_smodel, kd_trainset , val_dataloader, local_model_list):
    kd_lr = args.kd_lr
    kd_eval_batches_freq= args.kd_eval_batches_freq 
    kd_batch_size = args.kd_batch_size
    early_stopping_server_batches =args.early_stopping_server_batches
    total_n_server_pseudo_batches =args.total_n_server_pseudo_batches
    kd_KL_temperature =args.kd_KL_temperature

    
    optimizer_server_student = torch.optim.Adam(filter(lambda p: p.requires_grad, kd_server_smodel.parameters()),lr = kd_lr)
    scheduler_server_student = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_server_student,
        total_n_server_pseudo_batches,
        last_epoch=-1,
    )
    
    kd_model = kd_server_smodel
    # get the init server perf.
    kd_server_smodel.eval()
    _, init_perf_on_val = test(args.dataset, kd_server_smodel,val_dataloader )
    print("init_perf_on_val: ", init_perf_on_val)

    server_best_tracker = BestPerf(best_perf=None, larger_is_better=True)
    best_models = [None]
    validated_perfs = collections.defaultdict(list)
    ## note that kd_trainset is replaced to  kd_dataloader 
    distillation_data_loader=kd_trainset 
   
    print("use kd dataset", args.kd_dataset)   

    data_iter = iter(distillation_data_loader)
    batch_id= 0


    local_model_list = [model.cuda() for model in local_model_list]
    
    while batch_id < total_n_server_pseudo_batches:
        try:
            batch_data = next(data_iter)
        except StopIteration:
            data_iter = iter(distillation_data_loader)
            batch_data = next(data_iter)
        total_loss = 0 
        targets = batch_data[1].cuda()
        pseudo_data_student=batch_data[0].cuda() # 0 is data, 1 is label

        for cli_idx in range(len(local_model_list)):
            net = local_model_list[cli_idx]
            adv_input = adaad_inner_loss(net, kd_model, pseudo_data_student)
            net.train()
            optimizer_server_student.zero_grad()
            
            ori_outputs = net(pseudo_data_student)
            adv_outputs = net(adv_input)
            
            KL_loss = nn.KLDivLoss(reduction='none')
            
            with torch.no_grad():
                kd_model.eval()
                t_ori_outputs = kd_model(pseudo_data_student)
                t_adv_outputs = kd_model(adv_input)
            
            ## chexpert 맞춤 코드 수정
            if targets.dim() > 1:
                targets = targets.argmax(dim=1)
                targets = targets.long()
            
            Lambda = torch.zeros(pseudo_data_student.size(0)).cuda()
            t_misclassified_group = (
                        torch.argmax(t_ori_outputs, dim=1) != targets)  
            Lambda[t_misclassified_group] = 1.0
            num_lambda = Lambda.sum().item()

            ori_true_probs = torch.gather(t_ori_outputs, 1, targets.unsqueeze(1)).squeeze()
            ori_max_probs, ori_max_indices = torch.max(t_ori_outputs, dim=1)
            t_ori_outputs[t_misclassified_group, targets[t_misclassified_group]] = ori_max_probs[t_misclassified_group]
            t_ori_outputs[t_misclassified_group, ori_max_indices[t_misclassified_group]] = ori_true_probs[t_misclassified_group]
            
            correct_probs = torch.gather(t_adv_outputs, 1, targets.unsqueeze(1)).squeeze()
            max_probs, max_indices = torch.max((t_adv_outputs.scatter(1, targets.unsqueeze(1), -float('inf'))), dim=1)
            margin = correct_probs - max_probs
            mask = margin < 0
            t_adv_outputs[mask, targets[mask]] = max_probs[mask]
            t_adv_outputs[mask, max_indices[mask]] = correct_probs[mask]
            
            loss_clean = (1 / (num_lambda + 1e-10)) * torch.sum(
                Lambda * KL_loss(F.log_softmax(ori_outputs, dim=1), F.softmax(t_ori_outputs.detach(), dim=1)).sum(
                    dim=1))

            loss_adv = (1 / (len(adv_input) - num_lambda)) * torch.sum(
                (1 - Lambda) * KL_loss(F.log_softmax(adv_outputs, dim=1), F.softmax(t_adv_outputs.detach(), dim=1)).sum(dim=1))
            
            loss = (num_lambda / kd_batch_size) * loss_clean + ((kd_batch_size - num_lambda) / kd_batch_size) * loss_adv

            loss.backward()
            total_loss += loss.data
            optimizer_server_student.step()

        # after each batch.
        if scheduler_server_student is not None:
            scheduler_server_student.step()
        # overfit need early stop
        
        if (batch_id+1) % kd_eval_batches_freq == 0:
            kd_model.eval()
            _, validated_perf = test(args.dataset,
                kd_model, val_dataloader,log=False
            )
            log_str = ('Server Batch[{0:03}/{1:03}] '
                    'KD:{kd_loss:.4f} ValAcc{val_acc}'.format(
                    batch_id, total_n_server_pseudo_batches ,kd_loss= total_loss, val_acc =validated_perf   ))
            print(log_str)

              # check early stopping.
            if check_early_stopping(
                model=kd_model,
                model_ind=0,
                best_tracker=server_best_tracker,
                validated_perf=validated_perf,
                validated_perfs=validated_perfs,
                perf_index=batch_id + 1,
                early_stopping_batches=early_stopping_server_batches,
                best_models=best_models,
            ):
                break

        batch_id += 1

    use_init_server_model = (
            True
            if init_perf_on_val  >= server_best_tracker.best_perf
            else False
        )

        # get the server model.
    if use_init_server_model:
        print("use init server model instead.")
        return None, 0
    else:
        print("use distillation model at server step {} with val performance {}".format(server_best_tracker.get_best_perf_loc,server_best_tracker.best_perf  ))
        kd_model = best_models[0] 
        return kd_model, server_best_tracker.get_best_perf_loc -1 
