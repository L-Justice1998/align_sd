import torch.nn as nn
import torch

# class EMAModel(nn.Module):
#     def __init__(self, model, decay=0.999, bias_correction=False, num_updates=0, device=None):
#         super(EMAModel, self).__init__()
#         self.module = deepcopy(model).eval()
#         self.bias_correction = bias_correction
#         self.num_updates = num_updates
#         self.decay = decay
#         self.device = device
#         if self.device is not None:
#             self.module.to(device=self.device)

#     def _update(self, model, update_fn):
#         with torch.no_grad():
#             for ema_v, model_v in zip(self.module.state_dict().values(),
#                                       model.state_dict().values()):
#                 if self.device is not None:
#                     model_v = model_v.to(device=self.device)
#                 ema_v.copy_(update_fn(ema_v, model_v))

#     def update(self, model):
#         if self.bias_correction:
#             debias_term = (1 - self.decay ** self.num_updates)
#             self._update(model, update_fn=lambda e, m: (self.decay * e + (1 - self.decay) * m) / debias_term)
#             self.num_updates += 1
#         else:
#             self._update(model, update_fn=lambda e, m: self.decay * e + (1 - self.decay) * m)

#     def set(self, model):
#         self._update(model, update_fn=lambda e, m: m)
class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay=0.999, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)

