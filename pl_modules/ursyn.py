from torch import nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, List
# Local imports
from pl_modules.base import BaseModel
from losses.ursyn_loss import URSynLoss
from models.mmfusion import MMFusion

class URSyn(BaseModel):
    """Unique, redundant, and synergistic multimodal representation learning.

    URSyn encodes two augmented multimodal views and decomposes their
    representations into modality-specific unique factors, cross-modal redundant
    factors, and a joint synergistic factor.
    """

    def __init__(self,
                 encoder: MMFusion,
                 projection: nn.Module,
                 optim_kwargs: Dict,
                 loss_kwargs: Dict):
        """
        Args:
            encoder: Multi-modal fusion encoder
            projection: MLP projector to the latent space
            optim_kwargs: Optimization hyper-parameters
            loss_kwargs: Hyper-parameters for the URSyn loss.
        """
        super(URSyn, self).__init__(optim_kwargs)

        self.encoder = encoder

        self.head = projection

        self.loss = URSynLoss(**loss_kwargs)

    @staticmethod
    def _build_mlp(in_dim, mlp_dim, out_dim):
        return nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(in_dim, mlp_dim)),
            ("bn1", nn.SyncBatchNorm(mlp_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(mlp_dim, mlp_dim)),
            ("bn2", nn.SyncBatchNorm(mlp_dim)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(mlp_dim, out_dim)),
        ]))

    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor]):
        # compute features for all modalities
        all_masks = self.gen_all_possible_masks(len(x1))
        use_infmasking = (
            getattr(self.encoder, "infmasking_num_views", 0) > 0
            and getattr(self.loss, "lambda_infmask", 0.0) > 0
        )
        out1 = self.encoder(
            x1,
            mask_modalities=all_masks,
            return_infmasked_synergy=use_infmasking,
            return_ursyn_factors=True,
        )
        out2 = self.encoder(
            x2,
            mask_modalities=all_masks,
            return_infmasked_synergy=use_infmasking,
            return_ursyn_factors=True,
        )
        if use_infmasking:
            z1, z_U1, z_R1, z_S1, z_S_masked1 = out1
            z2, z_U2, z_R2, z_S2, z_S_masked2 = out2
        else:
            z1, z_U1, z_R1, z_S1 = out1
            z2, z_U2, z_R2, z_S2 = out2

        z1 = [self.head(z) for z in z1]
        z2 = [self.head(z) for z in z2]
        z_U1 = [self.head(z) for z in z_U1]
        z_R1 = [self.head(z) for z in z_R1]
        z_S1 = [self.head(z) for z in z_S1]
        z_U2 = [self.head(z) for z in z_U2]
        z_R2 = [self.head(z) for z in z_R2]
        z_S2 = [self.head(z) for z in z_S2]

        outputs = {'Z1': z1,
                'Z2': z2,
                'z_U1': z_U1,
                'z_R1': z_R1,
                'z_S1': z_S1,
                'z_U2': z_U2,
                'z_R2': z_R2,
                'z_S2': z_S2}
        if use_infmasking:
            outputs["z_S_masked1"] = [self.head(z) for z in z_S_masked1]
            outputs["z_S_masked2"] = [self.head(z) for z in z_S_masked2]
            trainer = getattr(self, "_trainer", None)
            current = float(getattr(
                trainer, "current_epoch", getattr(self, "custom_current_epoch", 0)
            ))
            total = max(float(getattr(
                trainer, "max_epochs", getattr(self, "total_epochs", 1)
            )), 1.0)
            outputs["training_progress"] = min(max((current + 1.0) / total, 0.0), 1.0)
        return outputs
    
    def gen_all_possible_masks(self, n_mod: int):
        """
        :param n_mod: int
        :return: a list of `n_mod` + 1 boolean masks [Mi] such that all but one bool are False.
            A last bool mask is added where all bool are True
        Examples:
        *   For n_mod==2:
            masks == [[True, False], [False, True], [True, True]]
        *   For n_mod == 3:
            masks == [[True, False, False], [False, True, False], [False, False, True], [True, True, True]]
        """
        masks = []
        for L in range(n_mod):
            mask = [s == L for s in range(n_mod)]
            masks.append(mask)
        masks.append([True for _ in range(n_mod)])
        return masks

    def extract_features(self, loader: torch.utils.data.DataLoader, **kwargs):
        X_all, y_all = [], []
        all_masks = None
        # `fusion_tau` is the routing temperature, not an encoder argument.
        kwargs = dict(kwargs)
        tau = kwargs.pop("fusion_tau", 0.5)

        for X_, y_ in loader:
            if isinstance(X_, torch.Tensor):
                X_ = [X_]

            X_ = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in X_]
            y_ = y_.to(self.device)

            if all_masks is None:
                all_masks = self.gen_all_possible_masks(len(X_))

            with torch.inference_mode():
                Z_out, U, R, S = self.encoder(X_, mask_modalities=all_masks, **kwargs)
                
                R0 = torch.stack(R, dim=0).mean(dim=0)
                S0 = S[0]

                Z0 = list(U) + [R0, S0]      # list of [B, D]
                Z = torch.stack(Z0, dim=1)        # [B, M, D]
                weights = self.compute_diversity_weights(Z0, tau=tau)  # [B, M]
                Z = Z * weights.unsqueeze(-1)
                feats = Z.reshape(Z.shape[0], -1)

            X_all.append(feats.cpu())
            y_all.append(y_.cpu())

        X_all = torch.cat(X_all, dim=0).to(self.device)
        y_all = torch.cat(y_all, dim=0).to(self.device)

        torch.cuda.empty_cache()
        return X_all, y_all

    def compute_diversity_weights(self, reprs, tau=0.5, eps=1e-8):
        Z = torch.stack(reprs, dim=1)  # [B, M, D]
        V = F.normalize(Z, p=2, dim=-1)
        sim = torch.bmm(V, V.transpose(1, 2))  # [B, M, M]

        M = Z.size(1)
        eye = torch.eye(M, device=Z.device, dtype=torch.bool).unsqueeze(0)
        sim = sim.masked_fill(eye, 0.0)

        alpha = -sim.sum(dim=-1)   # [B, M]
        w = torch.softmax(alpha / (tau + eps), dim=-1)
        return w

    def set_total_epochs(self, total_epochs: int):
        self.total_epochs = int(total_epochs)

    def set_current_epoch(self, current_epoch: int):
        self.custom_current_epoch = int(current_epoch)
