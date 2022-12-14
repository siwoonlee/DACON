import numpy as np
import torch


def create_masked_titles(title_vecs):
    device = title_vecs.device
    masks = []
    for title_vec in title_vecs:
        mask = np.random.choice(
            [True, False], size=len(title_vec), p=[0.75, 0.25]
        )
        masks.append(mask)
    return torch.tensor(masks, dtype=torch.long, device=device)
