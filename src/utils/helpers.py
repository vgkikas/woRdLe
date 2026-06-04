import torch

def build_action_mask(action_size, available_actions, device):
    """
    Builds a mask tensor for the available actions.
    """
    mask = torch.full((action_size,), -1e9, device=device)
    if available_actions:
        mask[available_actions] = 0.0
    return mask