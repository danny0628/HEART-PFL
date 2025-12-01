import copy
import numpy as np
import torch
import json
import os
from .base_al import FedBaseAL
import wandb
import torchattacks 
import torch.nn.functional as F
from torch.nn import CosineSimilarity
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR


class AL(FedBaseAL):
    """Base class for FL algos"""

    def __init__(
        self,
        args,
        clients,
        train_data,
        test_data,
        global_model,
        kd_trainset,
        val_dataloader,
        test_dataloader,
    ):
        if "fix" in args.net:  # only update the layers in "update_layer_name"
            if "adapter" in args.net:
                self.update_layer_name = [
                    "adapter",
                    "bn1",
                    "bn2",
                    "fc.weight",
                    "fc.bias",
                ]  # adapter layers, batch norm layers, fc layers
            elif "out" in args.net:
                self.update_layer_name = ["fc.weight", "fc.bias"]
            elif "inp" in args.net:
                self.update_layer_name = ["conv1.weight", "bn1.weight", "bn1.bias"]
            else:
                raise NotImplementedError

            for (
                n,
                p,
            ) in (
                global_model.named_parameters()
            ):  # fix other layers expect update_layer_name
                if (
                    self.is_global_update(n) == False
                ):  # fix the global layers; only finetune personalized layers
                    p.requires_grad_(False)

        super().__init__(
            args,
            clients,
            train_data,
            test_data,
            global_model,
            kd_trainset,
            val_dataloader,
            test_dataloader,
        )

        self.personalized_model_list = []
        self.personalized_optimizers = []
        self.personalized_privacy_engines = []

        for u in clients["train_users"]:
            per_model = copy.deepcopy(global_model)
            local_optim = torch.optim.SGD(per_model.parameters(), lr=args.lr)

            if (
                args.load_checkpoint is not None
                and "_fedavg_" not in args.load_checkpoint
            ):
                fname = os.path.join(args.load_checkpoint, "permodel_{}.ckpt".format(u))
                stat_dict = torch.load(fname)
                load_epoch = stat_dict["epoch"]
                per_model.load_state_dict(stat_dict["state_dict"], strict=False)
                print(
                    "load personalized model epoch {} from {}".format(
                        fname, load_epoch, fname
                    )
                )

            self.personalized_optimizers.append(local_optim)
            self.personalized_model_list.append(per_model)
            
            self.feature_fn = lambda m, x: m.extract_features(x)
            
            flat_params = torch.cat([p.data.view(-1) for p in global_model.parameters()])
            self.h_dict = {u: torch.zeros_like(flat_params).to("cuda") for u in range(20)}


    def is_global_update(self, name):
        for update_name in self.update_layer_name:
            if update_name in name:
                return True
        return False

    def server_distillation(
        self, init_model, kd_trainset, val_dataloader, select_model_list
    ):
        from aggregation.distillation import Distillation

        distill_model, distill_step = Distillation(
            self.args,
            copy.deepcopy(init_model),
            kd_trainset,
            val_dataloader,
            select_model_list,
        )
        if self.args.log_online:
            import wandb

            wandb.log(
                {"DistillStep": distill_step}, step=self.current_round, commit=False
            )
        if distill_model is not None:
            _, global_acc = self.test(distill_model, self.test_dataloader)
            print("global model after distillation", global_acc)
            return distill_model, global_acc
        else:
            return None, None

    def server_aggregation(
        self, select_model_list, num_train_samples, update_layer_name
    ):
        from aggregation.averaging import FedAveraging

        w_glob_avg = FedAveraging(
            [c.state_dict() for c in select_model_list],
            weights=[],
            update_layer_name=update_layer_name,
        )
        self.sever_load_state_dict(w_glob_avg)
        
        trainable_params = sum(p.numel() for p in self.global_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.global_model.parameters())

        print(f"서버 모델의 총 파라미터 수: {total_params:,}")
        print(f"서버 모델의 학습 가능한 파라미터 수: {trainable_params:,}")

        if (
            self.args.aggregation == "kd"
            and self.current_round < self.args.kd_max_round
        ):
            distill_model, global_acc = self.server_distillation(
                self.global_model,
                self.kd_trainset,
                self.val_dataloader,
                select_model_list,
            )
            if distill_model is not None:
                self.global_model = distill_model

    def run_local_updates(
        self, u, local_model, train_dataloader_u, num_epochs, optimizer
    ):
        per_model = self.personalized_model_list[u]
        per_optimizer = self.personalized_optimizers[u]  
        
        h_k = self.h_dict[u]
        
        u_train_per_loss, u_train_per_acc  = self.train_per(
            per_model,
            local_model,
            self.args.lmbda,
            train_dataloader_u,
            num_epochs,
            per_optimizer,
            u=u,
            h_k=h_k,
        )

        u_train_loss, u_train_acc = self.train(
            local_model, train_dataloader_u, num_epochs, optimizer, u=u
        )
        return u_train_loss, u_train_acc

    def model_dist_norm_var(self, model, prox_center, norm=2):
        model_params = [p for (n, p) in model.named_parameters()]
        return sum(
            torch.norm(v.reshape(-1) - v1.reshape(-1)) ** 2
            for (v, v1) in zip(model_params, prox_center)
        ## 스케쥴러 옵션 1,2
        # scheduler = StepLR(optimizer, step_size=2, gamma=0.5)  
        # scheduler = CosineAnnealingLR(optimizer, T_ma
        )

    def train_per(
        self, model, global_model, lmbda, trainloader, epoch, optimizer, log=True, u=0, h_k=None,
    ):

        
        criterion = self.criterion
        
        ## 옵티마이저 옵션 1,2
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr
        )

        ## 스케쥴러 옵션 1,2
        # scheduler = StepLR(optimizer, step_size=2, gamma=0.5)  
        # scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=self.args.lr*0.1)

        prox_center = [p.detach() for (n, p) in global_model.named_parameters()]

        model.zero_grad()
        model.train()
        # -------------------------------------------------------------
        # Step 1: Extract class-wise local prototypes
        # -------------------------------------------------------------
        from collections import defaultdict
        import torch.nn.functional as F

        # local_prototypes = defaultdict(list)
        local_prototypes_layer = [defaultdict(list) for _ in range(4)] ## multi layer prototype
        
        with torch.no_grad():
            for x_batch, y_batch in trainloader:
                x_batch, y_batch = x_batch.to("cuda"), y_batch.to("cuda")
                features = model.extract_features(x_batch)  # use local model’s base encoder
                features_per_layer = model.extract_features_all(x_batch)
                
                # for f, label in zip(features, y_batch):
                #     local_prototypes[label.item()].append(f)
                
                for layer_idx, feats in enumerate(features_per_layer):
                    for f, label in zip(feats, y_batch):
                        local_prototypes_layer[layer_idx][label.item()].append(f)


        # mean_prototypes = {}
        # for cls, feats in local_prototypes.items():
        #     mean_prototypes[cls] = torch.stack(feats).mean(dim=0)
        
        mean_prototypes_layer = []
        for layer_proto_dict in local_prototypes_layer:
            layer_mean = {}
            for cls, feats in layer_proto_dict.items():
                layer_mean[cls] = torch.stack(feats).mean(dim=0)
            mean_prototypes_layer.append(layer_mean)

        # -------------------------------------------------------------s
        # Step 2: Local Training with FedProx + Prototype Alignment
        # -------------------------------------------------------------
        align_lambda = 0.1
        def train_core(data_loader, model, epoch, optimizer):
            for ep in range(epoch):
                train_loss = 0
                correct = 0
                total = 0

                for batch_idx, (x, y) in enumerate(data_loader):
                    x, y = x.to("cuda"), y.to("cuda")
                    optimizer.zero_grad()

                    pred = model(x)
                    ce_loss = criterion(pred, y)
                    prox_loss = lmbda * self.model_dist_norm_var(model, prox_center)

                    student_feats_all = model.extract_features_all(x)
                    with torch.no_grad():
                        # global_features = global_model.extract_features(x)
                        global_feats_all = global_model.extract_features_all(x)

                    align_loss = 0.0
                    cos_sim = CosineSimilarity(dim=1)
                    
                    ### per layer each reguliaze implement
                    for layer_idx in range(4):
                        student_feats = student_feats_all[layer_idx]  # shape: [B, D_l]
                        global_feats = global_feats_all[layer_idx]    # shape: [B, D_l]
                        proto_dict = mean_prototypes_layer[layer_idx]

                        for i in range(len(y)):
                            label = y[i].item()
                            if label not in proto_dict:
                                continue

                            target_proto = proto_dict[label]
                            g_feat = global_feats[i]
                            s_feat = student_feats[i]

                            if layer_idx in [0, 1]:  # low-level: cosine
                                sim = cos_sim(g_feat.unsqueeze(0), target_proto.unsqueeze(0))
                                loss = 1 - sim.mean()
                            else:  # high-level: MSE
                                loss = F.mse_loss(g_feat, target_proto)

                            align_loss += loss
                            # align_loss += F.mse_loss(g_feat, target_proto)

                    align_loss /= (len(y) * 4)  # 평균 정규화       
           
                    loss = ce_loss + prox_loss + align_lambda * align_loss
                    
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    total += y.size(0)
                    if self.args.dataset == "chexpert":
                        pred_c = (torch.sigmoid(pred) > 0.5).float()
                        correct += (pred_c == y).float().mean().item() * y.size(0)
                    else:
                        _, pred_c = pred.max(1)
                        correct += pred_c.eq(y).sum().item()

                print(
                    "personalized cli %d ep %d batch %d  train Loss: %.3f | Acc:%.3f%% | total %d"
                    % (
                        u,
                        ep,
                        batch_idx,
                        train_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        total,
                    )
                )
                # scheduler.step()
            
            acc = 100.0 * correct / total
            return train_loss / len(data_loader), acc

        _loss, _acc = train_core(trainloader, model, epoch, optimizer)

        # ============================================================
        # Loss Landscape Visualization for Average Client Loss (주석 처리됨)
        # 여러 client의 평균 loss로 landscape를 그립니다
        # 특정 조건에서만 실행하도록 설정 (예: 첫 번째 client이고 특정 라운드)
        # ============================================================
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # 조건: 첫 번째 client일 때만 실행 (라운드당 한 번)
        # u == 0 조건이 필요한 이유: train_per는 각 client마다 호출되지만,
        # loss landscape는 라운드당 한 번만 그려야 하므로
        # 실제로는 모든 client의 데이터를 사용합니다 (아래 for client_idx 루프 참고)
        if u == 0 and hasattr(self, 'current_round'):
            print(f"Computing loss landscape for all {len(self.personalized_model_list)} clients at round {self.current_round}...")
            
            # 1. Loss landscape 파라미터 설정
            num_points = 15  # grid 해상도 (15x15)
            alpha_range = np.linspace(-0.5, 0.5, num_points)
            beta_range = np.linspace(-0.5, 0.5, num_points)
            
            # 2. 기준 모델 (global_model)의 파라미터 저장
            original_params = [p.clone().detach() for p in global_model.parameters()]
            
            # 3. Filter-wise normalized 랜덤 방향 벡터 생성
            direction1 = []
            direction2 = []
            for p in global_model.parameters():
                d1 = torch.randn_like(p)
                d2 = torch.randn_like(p)
                
                if len(p.shape) >= 2:  # Conv, Linear 레이어
                    d1_norm = torch.norm(d1.view(p.shape[0], -1), dim=1, keepdim=True)
                    d2_norm = torch.norm(d2.view(p.shape[0], -1), dim=1, keepdim=True)
                    d1 = d1 / (d1_norm.view(-1, *([1]*(len(p.shape)-1))) + 1e-10)
                    d2 = d2 / (d2_norm.view(-1, *([1]*(len(p.shape)-1))) + 1e-10)
                else:  # Bias, BN parameters
                    d1 = d1 / (torch.norm(d1) + 1e-10)
                    d2 = d2 / (torch.norm(d2) + 1e-10)
                
                # 파라미터 크기에 비례하도록 스케일링
                d1 = d1 * torch.norm(p) * 0.1
                d2 = d2 * torch.norm(p) * 0.1
                
                direction1.append(d1)
                direction2.append(d2)
            
            # 4. 모든 client에 대한 평균 loss landscape 계산
            loss_surface = np.zeros((num_points, num_points))
            num_clients = len(self.personalized_model_list)
            
            for i, alpha in enumerate(alpha_range):
                for j, beta in enumerate(beta_range):
                    total_loss = 0.0
                    valid_clients = 0
                    
                    # 각 client에 대해 loss 계산
                    for client_idx in range(num_clients):
                        client_model = self.personalized_model_list[client_idx]
                        client_dataloader = self.train_data[client_idx]["dataloader"]
                        
                        # 현재 그리드 포인트로 모델 파라미터 이동
                        with torch.no_grad():
                            for p, orig_p, d1, d2 in zip(client_model.parameters(), original_params, direction1, direction2):
                                p.copy_(orig_p + alpha * d1 + beta * d2)
                        
                        # 해당 위치에서의 loss 계산
                        client_model.eval()
                        temp_loss = 0
                        temp_count = 0
                        
                        with torch.no_grad():
                            for batch_idx, (x, y) in enumerate(client_dataloader):
                                x, y = x.to('cuda'), y.to('cuda')
                                pred = client_model(x)
                                loss = criterion(pred, y)
                                
                                # NaN 체크
                                if not torch.isnan(loss) and not torch.isinf(loss):
                                    temp_loss += loss.item()
                                    temp_count += 1
                                
                                
                                # 계산 시간 절약을 위해 일부 배치만 사용
                                if batch_idx >= 5:  # 첫 5개 배치만
                                    break
                        
                        if temp_count > 0:
                            total_loss += temp_loss / temp_count
                            valid_clients += 1
                        
                        # 모델 파라미터를 원래 위치로 복원
                        with torch.no_grad():
                            client_state = self.personalized_model_list[client_idx].state_dict()
                            client_model.load_state_dict(client_state)
                    
                    # 모든 client의 평균 loss
                    if valid_clients > 0:
                        loss_surface[i, j] = total_loss / valid_clients
                    else:
                        loss_surface[i, j] = np.nan
                    
                    if (i * num_points + j + 1) % 10 == 0:
                        print(f"Progress: {(i * num_points + j + 1)}/{num_points*num_points} - "
                              f"alpha={alpha:.2f}, beta={beta:.2f}, avg_loss={loss_surface[i, j]:.4f}")
            
            # 5. NaN 값 처리 (interpolation)
            if np.isnan(loss_surface).any():
                print(f"Warning: {np.isnan(loss_surface).sum()} NaN values detected. Interpolating...")
                from scipy.interpolate import griddata
                mask = ~np.isnan(loss_surface)
                if mask.sum() > 3:
                    points = np.array(np.where(mask)).T
                    values = loss_surface[mask]
                    grid_x, grid_y = np.meshgrid(range(num_points), range(num_points))
                    loss_surface = griddata(points, values, (grid_x, grid_y), method='cubic', fill_value=values.mean())
            
            # 6. Loss landscape 시각화
            X, Y = np.meshgrid(alpha_range, beta_range)
            
            # Loss 범위를 0~20으로 통일
            vmin, vmax = 0, 20
            
            # 3D surface plot
            fig_3d = plt.figure(figsize=(10, 8))
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            surf = ax_3d.plot_surface(X, Y, loss_surface.T, cmap='viridis', alpha=0.8, edgecolor='none', vmin=vmin, vmax=vmax)
            ax_3d.set_xlabel('Direction 1 (α)', fontsize=12)
            ax_3d.set_ylabel('Direction 2 (β)', fontsize=12)
            ax_3d.set_zlabel('Average Loss', fontsize=12)
            ax_3d.set_zlim(vmin, vmax)  # Z축 범위 고정
            ax_3d.set_title(f'Average of {num_clients} Clients', fontsize=14, fontweight='bold')
            fig_3d.colorbar(surf, ax=ax_3d, shrink=0.6, aspect=10)
            plt.tight_layout()
            plt.savefig(f'/home/mjkim/WACV_2026/landscape_fig/ours/3d_adv_{self.current_round}.png', dpi=300, bbox_inches='tight')
            print(f"✓ 3D loss landscape saved to 'loss_landscape_clients_3d_round{self.current_round}.png'")
            plt.close(fig_3d)
        # ============================================================
        # End of Loss Landscape Visualization Code
        # ============================================================

        return _loss, _acc
    

    # def train_per(
    #     self, model, global_model, lmbda, trainloader, epoch, optimizer, log=True, u=0, h_k=None,
    # ):
    #     criterion = self.criterion
        
    #     ## 옵티마이저 옵션 1,2
    #     optimizer = torch.optim.SGD(
    #         filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr
    #     )

    #     prox_center = [p.detach() for (n, p) in global_model.named_parameters()]

    #     model.zero_grad()
    #     model.train()
        
    #     # -------------------------------------------------------------
    #     # Step 1: Extract class-wise local prototypes
    #     # -------------------------------------------------------------
    #     from collections import defaultdict
    #     import torch.nn.functional as F
    #     from torch.nn import CosineSimilarity

    #     local_prototypes_layer = [defaultdict(list) for _ in range(4)]
        
    #     with torch.no_grad():
    #         for x_batch, y_batch in trainloader:
    #             x_batch, y_batch = x_batch.to("cuda"), y_batch.to("cuda")
    #             features_per_layer = model.extract_features_all(x_batch)
                
    #             for layer_idx, feats in enumerate(features_per_layer):
    #                 for f, label in zip(feats, y_batch):
    #                     local_prototypes_layer[layer_idx][label.item()].append(f)

    #     mean_prototypes_layer = []
    #     for layer_proto_dict in local_prototypes_layer:
    #         layer_mean = {}
    #         for cls, feats in layer_proto_dict.items():
    #             layer_mean[cls] = torch.stack(feats).mean(dim=0)
    #         mean_prototypes_layer.append(layer_mean)

    #     # -------------------------------------------------------------
    #     # Step 2: Local Training with measurements
    #     # -------------------------------------------------------------
    #     align_lambda = 0.1
        
    #     # 측정 변수들 초기화
    #     total_directional_alignment = [0.0] * 4
    #     total_magnitude_diversity = [0.0] * 4
    #     total_samples = 0
        
    #     for ep in range(epoch):
    #         train_loss = 0
    #         correct = 0
    #         total = 0

    #         for batch_idx, (x, y) in enumerate(trainloader):
    #             x, y = x.to("cuda"), y.to("cuda")
    #             optimizer.zero_grad()

    #             pred = model(x)
    #             ce_loss = criterion(pred, y)
    #             prox_loss = lmbda * self.model_dist_norm_var(model, prox_center)

    #             student_feats_all = model.extract_features_all(x)
    #             with torch.no_grad():
    #                 global_feats_all = global_model.extract_features_all(x)

    #             align_loss = 0.0
    #             cos_sim = CosineSimilarity(dim=1)
                
    #             # 배치별 측정값들
    #             batch_directional_alignment = [0.0] * 4
    #             batch_magnitude_diversity = [0.0] * 4
                
    #             ### per layer each regularize + measurement
    #             for layer_idx in range(4):
    #                 student_feats = student_feats_all[layer_idx]
    #                 global_feats = global_feats_all[layer_idx]
    #                 proto_dict = mean_prototypes_layer[layer_idx]

    #                 # Feature Magnitude Diversity 계산
    #                 feature_magnitudes = torch.norm(global_feats, dim=-1)
    #                 magnitude_std = torch.std(feature_magnitudes).item()
    #                 batch_magnitude_diversity[layer_idx] = magnitude_std
                    
    #                 # Directional Alignment Score 계산
    #                 layer_directional_scores = []

    #                 for i in range(len(y)):
    #                     label = y[i].item()
    #                     if label not in proto_dict:
    #                         continue

    #                     target_proto = proto_dict[label]
    #                     g_feat = global_feats[i]

    #                     # Directional Alignment Score 측정
    #                     g_feat_norm = F.normalize(g_feat.unsqueeze(0), dim=-1)
    #                     proto_norm = F.normalize(target_proto.unsqueeze(0), dim=-1)
    #                     directional_sim = cos_sim(g_feat_norm, proto_norm).item()
    #                     layer_directional_scores.append(directional_sim)

    #                     # 기존 alignment loss 계산
    #                     if layer_idx in [0, 1]:  # low-level: cosine
    #                         sim = cos_sim(g_feat.unsqueeze(0), target_proto.unsqueeze(0))
    #                         loss = 1 - sim.mean()
    #                     else:  # high-level: MSE
    #                         loss = F.mse_loss(g_feat, target_proto)

    #                     align_loss += loss

    #                 # 레이어별 평균 directional alignment 계산
    #                 if layer_directional_scores:
    #                     batch_directional_alignment[layer_idx] = sum(layer_directional_scores) / len(layer_directional_scores)

    #             align_loss /= (len(y) * 4)
                
    #             # 전체 측정값에 누적
    #             for layer_idx in range(4):
    #                 total_directional_alignment[layer_idx] += batch_directional_alignment[layer_idx] * len(y)
    #                 total_magnitude_diversity[layer_idx] += batch_magnitude_diversity[layer_idx] * len(y)
    #             total_samples += len(y)
    
    #             loss = ce_loss + prox_loss + align_lambda * align_loss
                
    #             loss.backward()
    #             optimizer.step()

    #             train_loss += loss.item()
    #             total += y.size(0)
    #             if self.args.dataset == "chexpert":
    #                 pred_c = (torch.sigmoid(pred) > 0.5).float()
    #                 correct += (pred_c == y).float().mean().item() * y.size(0)
    #             else:
    #                 _, pred_c = pred.max(1)
    #                 correct += pred_c.eq(y).sum().item()

    #         print(
    #             "personalized cli %d ep %d batch %d  train Loss: %.3f | Acc:%.3f%% | total %d"
    #             % (
    #                 u,
    #                 ep,
    #                 batch_idx,
    #                 train_loss / (batch_idx + 1),
    #                 100.0 * correct / total,
    #                 total,
    #             )
    #         )
        
    #     # 최종 평균 계산
    #     avg_directional_alignment = [score / total_samples if total_samples > 0 else 0.0 for score in total_directional_alignment]
    #     avg_magnitude_diversity = [score / total_samples if total_samples > 0 else 0.0 for score in total_magnitude_diversity]
        
    #     acc = 100.0 * correct / total
        
    #     # 측정 결과 출력
    #     if log:
    #         print(f"Client {u} - Directional Alignment: {[f'{score:.4f}' for score in avg_directional_alignment]}")
    #         print(f"Client {u} - Magnitude Diversity: {[f'{score:.4f}' for score in avg_magnitude_diversity]}")
        
    #     return train_loss / len(trainloader), acc, avg_directional_alignment, avg_magnitude_diversity


    def local_finetune_one_round(self, current_round):
        self.current_round = current_round

        (
            train_acces,
            num_train_samples,
        ) = ([], [])
        self.selected_clients = list(range(len(self.train_users)))
        for u in self.selected_clients:
            train_dataloader_u = self.train_data[u]["dataloader"]
            per_model = self.personalized_model_list[u]
            per_optim = self.personalized_optimizers[u]
            # update local model
            u_train_loss, u_train_acc = self.run_local_updates(
                u, per_model, train_dataloader_u, 1, per_optim
            )

            
            num_train_samples.append(len(self.train_data[u]["indices"]))
            train_acces.append(u_train_acc)

        if self.args.log_online:

            wandb.log(
                {
                    "trainUserLocalAcc": (
                        np.nan
                        if len(train_acces) == 0
                        else np.average(train_acces, weights=num_train_samples)
                    ),
                },
                step=self.current_round,
            )

        if self.current_round % self.args.eval_every == 0:
            self.save_checkpoints(-1)
        

    def save_checkpoints(self, global_acc):

        for u in self.selected_clients:
            test_dataloader_u = self.test_data[u]["dataloader"]
            per_model = self.personalized_model_list[u]
            u_test_per_loss, u_test_per_acc = self.test(
                per_model, test_dataloader_u
            )  # test with local testdataset
            self.saved_test_per_acces[u] = u_test_per_acc
            
            print(f'{u} client test accuracy : {u_test_per_acc}')

            u_test_per_cent_loss, u_test_per_cent_acc = self.test(
                per_model, self.test_dataloader
            )  # test with centralized testdataset

            self.saved_test_per_cent_acces[u] = u_test_per_cent_acc

            # if self.args.nologging == False:
            #     if u_test_per_acc > self.per_best_acc[u]:
            #         self.per_best_acc[u] = u_test_per_acc
            #         torch.save(
            #             {
            #                 "state_dict": per_model.state_dict(),
            #                 "epoch": self.current_round,
            #             },
            #             os.path.join(
            #                 self.args.output_summary_file, "permodel_{}.ckpt".format(u)
            #             ),
            #         )

        if self.args.log_online:

            wandb.log(
                {
                    "testUserPerLocalAcc": (
                        np.nan
                        if len(self.saved_test_per_acces) == 0
                        else np.average(
                            self.saved_test_per_acces,
                            weights=self.num_test_samples_all_clients,
                        )
                    ),
                    "testUserPerCentAcc": (
                        np.nan
                        if len(self.saved_test_per_cent_acces) == 0
                        else np.average(self.saved_test_per_cent_acces)
                    ),
                },
                step=self.current_round,
            )

        if self.args.nologging == False:

            self.saved_results["round"].append(self.current_round)
            self.saved_results["testServerCentAcc"].append(global_acc)
            self.saved_results["testUserPerLocalAcc"].append(
                np.average(
                    self.saved_test_per_acces, weights=self.num_test_samples_all_clients
                )
            )
            self.saved_results["testUserPerLocalAccStd"].append(
                round(np.array(self.saved_test_per_acces).std(), 5)
            )
            self.saved_results["testUserPerCentAcc"].append(
                np.average(self.saved_test_per_cent_acces)
            )
            self.saved_results["testUserPerCentAccStd"].append(
                round(np.array(self.saved_test_per_cent_acces).std(), 5)
            )

            with open(
                os.path.join(self.args.output_summary_file, "results.json"), "w"
            ) as f:
                json.dump(self.saved_results, f)

            # check to save the model
            # if global_acc > self.global_best_acc:
            #     self.global_best_acc = global_acc
            #     torch.save(
            #         {
            #             "state_dict": self.global_model.state_dict(),
            #             "epoch": self.current_round,
            #         },
            #         os.path.join(self.args.output_summary_file, "gmodel.ckpt"),
            #     )

            wandb.log(
                {
                    "testUserPerLocalAcc": (
                        np.nan
                        if len(self.saved_test_per_acces) == 0
                        else np.average(
                            self.saved_test_per_acces,
                            weights=self.num_test_samples_all_clients,
                        )
                    ),
                },
                step=self.current_round,
            )
