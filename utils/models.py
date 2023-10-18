import importlib
import torch



def load_model(config, device):
    if config['jit']:
        net = torch.jit.load(config['weights'])
    else:
        module = importlib.import_module(config["module"])
        net = getattr(module, config["class_name"])(**vars(config["args"]))
        net.load_state_dict(torch.load(config["weights"], map_location='cpu'))
    net.to(device)
    net.eval()
    return net
    