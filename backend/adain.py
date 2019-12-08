import torch
from torchvision import transforms
from model import Model
import sys

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

class adain :
    def __init__(self, model_path, style_tensor, device='cpu') :
        model = Model()
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        model = model.to(device)
        self.device = device
        self.model = model
        self.style = normalize(style_tensor).unsqueeze(0).to(device)

    # Tensor -> Tensor
    def __call__(self, contents) :
        ret = []
        with torch.no_grad() :
            for content_tensor in contents :
                c_tensor = normalize(content_tensor).unsqueeze(0).to(self.device)
                result = self.model.generate(c_tensor, self.style)
                ret.append(denorm(result, self.device).squeeze(0).to('cpu'))
        return ret
