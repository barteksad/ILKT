import torch


def stiffness(gradient_1, gradient_2, s_type="cosine"):
    # gradient_1.shape == (d_model)
    # gradient_2.shape == (d_model)

    if s_type == "cosine":
        return torch.nn.functional.cosine_similarity(
            gradient_1.view(1, -1), gradient_2.view(1, -1)
        )
    elif s_type == 'sign':
        return torch.sign(gradient_1) * torch.sign(gradient_2)
