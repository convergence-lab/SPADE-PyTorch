import torch
import torch.nn as nn

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super().__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real, generator=False):
        target_tensor = None
        b = input.shape[0]
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = torch.tensor(real_tensor)
            if generator:
                target_tensor = self.real_label_var
            else:
                target_tensor = self.real_label_var #- (torch.rand(self.real_label_var.shape) * 0.5 - 0.25)
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = torch.tensor(fake_tensor)
            target_tensor = self.fake_label_var #+ torch.rand(self.fake_label_var.shape) * 0.3
        return target_tensor

    def __call__(self, input, target_is_real, generator=False):        
        target_tensor = self.get_target_tensor(input, target_is_real, generator)
        return self.loss(input, target_tensor.to(torch.device('cuda')))