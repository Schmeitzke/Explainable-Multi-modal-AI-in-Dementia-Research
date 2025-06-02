import torch
import torch.nn as nn
import torch.nn.functional as F
from MMDF import config

class SelfAdaptiveLoss(nn.Module):
    """
    Combines Symmetric Cross-Entropy (SCE) and Enhanced Focal Loss (EFL) with adaptive weights.
    Implements the self-adaptive mechanism described in equations 17-20 of the paper.
    """
    def __init__(self):
        super(SelfAdaptiveLoss, self).__init__()
        alpha = config.loss_sce_alpha
        beta = config.loss_sce_beta
        gamma = config.loss_efl_gamma
        self.adaptive_alpha = config.loss_adaptive_weight_alpha
        
        w1 = config.loss_initial_weight1
        w2 = config.loss_initial_weight2
        
        self.w1 = nn.Parameter(torch.tensor(w1, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.tensor(w2, dtype=torch.float32))
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = 1e-6
        
        self.initial_losses = {'sce': None, 'efl': None}
        self.loss_history = {'sce': [], 'efl': []}

    def calculate_ce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, reduction='mean')

    def calculate_rce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=prob.size(1)).float()
        rce = torch.sum(one_hot * torch.log(prob + self.eps), dim=1)
        return rce.mean()

    def calculate_sce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.calculate_ce(logits, targets)
        rce = self.calculate_rce(logits, targets)
        return self.alpha * ce + self.beta * rce

    def calculate_efl(self,
                      logits: torch.Tensor,
                      targets: torch.Tensor,
                      class_weights: torch.Tensor = None) -> torch.Tensor:
        prob = F.softmax(logits, dim=1)
        pt = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        modulation = (1 - pt) ** self.gamma
        ce_loss = -torch.log(pt + self.eps)
        
        if class_weights is None:
            weight = 1.0
        else:
            weight = class_weights[targets]
        
        loss = weight * modulation * ce_loss
        return loss.mean()

    def calculate_gradient_norm(self, loss, model_params):
        """Calculate L2 norm of gradients for the loss."""
        grads = torch.autograd.grad(loss, model_params, retain_graph=True, create_graph=True)
        grad_norm = torch.sqrt(sum(torch.sum(grad ** 2) for grad in grads))
        return grad_norm

    def update_weights(self,
                       loss_sce: torch.Tensor,
                       loss_efl: torch.Tensor,
                       epoch: int,
                       model_params=None) -> None:
        """
        Update weights according to equations (18-20) in the paper.
        """
        if self.initial_losses['sce'] is None:
            self.initial_losses['sce'] = loss_sce.detach()
            self.initial_losses['efl'] = loss_efl.detach()
            return
        
        current_sce_ratio = loss_sce / (self.initial_losses['sce'] + self.eps)
        current_efl_ratio = loss_efl / (self.initial_losses['efl'] + self.eps)
        
        self.loss_history['sce'].append(current_sce_ratio.item())
        self.loss_history['efl'].append(current_efl_ratio.item())
        
        if len(self.loss_history['sce']) > 10:
            self.loss_history['sce'] = self.loss_history['sce'][-10:]
            self.loss_history['efl'] = self.loss_history['efl'][-10:]
                        
        total = loss_sce + loss_efl + self.eps
        new_w1 = loss_sce / total
        new_w2 = loss_efl / total
        
        momentum = 0.9
        with torch.no_grad():
            self.w1.data = momentum * self.w1.data + (1 - momentum) * new_w1
            self.w2.data = momentum * self.w2.data + (1 - momentum) * new_w2

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                epoch: int = 0,
                class_weights: torch.Tensor = None,
                model_params=None) -> torch.Tensor:
        loss_sce = self.calculate_sce(logits, targets)
        loss_efl = self.calculate_efl(logits, targets, class_weights)
        
        self.update_weights(loss_sce, loss_efl, epoch, model_params)
        
        total_loss = self.w1 * loss_sce + self.w2 * loss_efl
        return total_loss
