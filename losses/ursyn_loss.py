import torch.nn.functional as func
import torch
import torch.nn as nn
from utils import all_gather_batch_with_grad
#local
from losses.cos_similarity import Cosine_loss
from losses.entropy import entropy_gradeint


class URSynLoss(nn.Module):

    def __init__(self, temperature=0.1, weights=None, entropy_weight=0.01,
                 epsilon=0.2, align_weight=1.0, lambda_repel=50,
                 lambda_R=1, lambda_UR_dis=1, lambda_S=1,
                 lambda_S_dis=1):
        super().__init__()
        self.INF = 1e8
        self.weights = weights
        self.temperature = temperature
        self.Cosine_loss = Cosine_loss(margin=epsilon)
        self.entropy_weight = entropy_weight
        self.align_weight = align_weight
        self.lambda_repel = lambda_repel
        self.lambda_R = lambda_R
        self.lambda_UR_dis = lambda_UR_dis
        self.lambda_S = lambda_S
        self.lambda_S_dis = lambda_S_dis

    def infonce(self, z1, z2):
        N = len(z1)
        sim_zii = (z1 @ z1.T) / self.temperature  # [N, N]
        sim_zjj = (z2 @ z2.T) / self.temperature  # [N, N]
        sim_zij = (z1 @ z2.T) / self.temperature  # [N, N]
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z1.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z1.device)
        sim_Z = torch.cat([
            torch.cat([sim_zij, sim_zii], dim=1),
            torch.cat([sim_zjj, sim_zij.T], dim=1)], dim=0)
        log_sim_Z = func.log_softmax(sim_Z, dim=1)
        loss = - torch.diag(log_sim_Z).mean()
        # compute SSL accuracy
        with torch.no_grad():
            pred = torch.argmax(sim_zij, dim=1)
            correct = pred.eq(torch.arange(N, device=z1.device)).sum()
            acc = 100 * correct / N
        return loss, acc

    def multi_positive_infonce(self, anchor, *positives):
        """InfoNCE with several positives: `anchor[i]` must match `positives[j][i]`.

        Args:
            anchor: (B, D) query representations
            *positives: several (B, D) positive representations

        Returns:
            loss: scalar
            acc: scalar (percentage)
        """
        losses = []
        accs = []

        for positive in positives:
            loss, acc = self.infonce(anchor, positive)
            losses.append(loss)
            accs.append(acc)
        
        avg_loss = torch.mean(torch.stack(losses))
        avg_acc = torch.mean(torch.stack(accs))
        
        return avg_loss, avg_acc

    def forward(self, outputs):
        # keep x0 away from inputs 
        def dis_tangle(x0, x1, *inputs):
            n = len(inputs)
            align = torch.einsum("nc,nc->n", x0, Z2[0]).mean()
            align += torch.einsum("nc,nc->n", x1, Z1[0]).mean()
            entropy_term = entropy_gradeint(x0)
            entropy_term += entropy_gradeint(x1)
            repel = 0
            n1 = int(n / 2)
            for i in range(n1): 
                repel += self.Cosine_loss(x0, inputs[i])
                repel += self.Cosine_loss(x1, inputs[i + n1])
            if n1 > 0:
                repel = repel / (2.0 * n1)
            loss = - self.entropy_weight * entropy_term - self.align_weight * align + self.lambda_repel * repel
            return loss
        def dis_tangle_UR(x0, x1, *inputs):
            n = len(inputs)
            entropy_term = entropy_gradeint(x0)
            entropy_term += entropy_gradeint(x1)
            repel = 0
            n1 = int(n / 2)
            for i in range(n1): 
                repel += self.Cosine_loss(x0, inputs[i])
                repel += self.Cosine_loss(x1, inputs[i + n1])
            if n1 > 0:
                repel = repel / (2.0 * n1)
            loss = - self.entropy_weight * entropy_term + self.lambda_repel * repel
            return loss
        

        z1, z2 = outputs["Z1"], outputs["Z2"]
        z_U1, z_R1, z_S1 = outputs["z_U1"], outputs["z_R1"], outputs["z_S1"]
        z_U2, z_R2, z_S2 = outputs["z_U2"], outputs["z_R2"], outputs["z_S2"]

        feats = z1 + z2 + z_U1 + z_R1 + z_S1 + z_U2 + z_R2 + z_S2
        feats = [func.normalize(z, p=2, dim=-1) for z in feats]
        feats = all_gather_batch_with_grad(feats)

        n = len(z1) - 1
        z1 = feats[:n]                          # [z1_v, z1_t]
        Z1 = feats[n: n + 1]                    # [z1_vt]
        z2 = feats[n + 1: 2 * n + 1]            # [z2_v, z2_t]
        Z2 = feats[2 * n + 1: 2 * n + 2]        # [z2_vt]
        z_U1 = feats[2 * n + 2: 3 * n + 2]      # Unique;  [u1_v, u1_t]
        z_R1 = feats[3 * n + 2: 4 * n + 2]      # Redundant; [r1_v, r1_t]
        z_S1 = feats[4 * n + 2: 4 * n + 3]      # Synergy; [s1_vt]
        z_U2 = feats[4 * n + 3: 5 * n + 3]      # Unique;  [u2_v, u2_t]
        z_R2 = feats[5 * n + 3: 6 * n + 3]      # Redundant; [r2_v, r2_t]
        z_S2 = feats[6 * n + 3:]                # Synergy; [s2_vt]

        # Uniqueness: match each unimodal view with the unique factor of the other view.
        loss_U = []
        acc_U = []
        # Redundancy: match each unimodal view with all redundant factors of the other view.
        loss_R = []
        acc_R = []

        for i in range(n):
            loss, acc = self.infonce(z1[i], z_U2[i])
            loss2, acc2 = self.infonce(z2[i], z_U1[i])
            loss3 = (loss + loss2) / 2
            acc3 = (acc + acc2) / 2
            loss_U.append(loss3)
            acc_U.append(acc3)
            # Zi R1' R2' ... Rn' and Zi' R1 R2 ... Rn
            loss, acc = self.multi_positive_infonce(z1[i], *z_R2)
            loss2, acc2 = self.multi_positive_infonce(z2[i], *z_R1)
            loss3 = (loss + loss2) / 2
            acc3 = (acc + acc2) / 2
            loss_R.append(loss3)
            acc_R.append(acc3)
        
        # Align redundant factors of different modalities with each other.
        for i in range(n):
            for j in range(i + 1, n):
                loss, acc = self.infonce(z_R1[i],z_R1[j])
                loss2, acc2 = self.infonce(z_R2[i], z_R2[j])
                loss3 = (loss + loss2) / 2
                acc3 = (acc + acc2) / 2
                loss_R.append(loss3)
                acc_R.append(acc3)

        # Disentangle unique from redundant factors.
        loss_UR_dis = []
        for i in range(n):
            lossUR_dis = dis_tangle_UR(z_U1[i], z_U2[i], *z_R1, *z_R2)
            loss_UR_dis.append(lossUR_dis)

        # Synergy: align the joint synergistic factors across views.
        loss_S = []
        acc_S = []
        loss, acc = self.infonce(z_S1[0], z_S2[0])
        loss_S.append(loss)
        acc_S.append(acc)

        # ... and keep them away from the unimodal factors.
        lossS_dis = dis_tangle(z_S1[0], z_S2[0], *z1, *z2)

        lossU = torch.mean(torch.stack(loss_U))
        lossR = torch.mean(torch.stack(loss_R)) * self.lambda_R
        lossUR_dis = torch.mean(torch.stack(loss_UR_dis)) * self.lambda_UR_dis
        lossS = torch.mean(torch.stack(loss_S)) * self.lambda_S
        lossS_dis = lossS_dis * self.lambda_S_dis

        accU = torch.mean(torch.stack(acc_U))
        accR = torch.mean(torch.stack(acc_R))
        accS = torch.mean(torch.stack(acc_S))

        loss_list = [lossU, lossR, lossUR_dis, lossS, lossS_dis]
        loss_vec = torch.stack(loss_list)  
        acc_list = [accU, accR, accS]

        if self.weights is not None:
            weights = torch.tensor(self.weights, device=z1[0].device, dtype=loss_list[0].dtype)
            loss_vec = loss_vec* weights
        
        loss = torch.sum(loss_vec)
        ssl_acc_dict = {"ssl_acc_%i" % i: acc_ for i, acc_ in enumerate(acc_list)}
        losses_dict = {f"ssl_loss_w_{i}": loss_vec[i].detach() for i in range(5)}

        acc = torch.mean(torch.stack(acc_list))

        return {"loss": loss, "ssl_acc": acc, **ssl_acc_dict, **losses_dict}

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)
