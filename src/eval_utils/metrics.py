import torch

def stiffness(gradient_1, gradient_2, s_type="cosine"):
    if s_type == "cosine":
        gradient_1 = gradient_1/(torch.norm(gradient_1, dim=-1, p=2).unsqueeze(-1) + 1e-6)
        gradient_2 = gradient_2/(torch.norm(gradient_2, dim=-1, p=2).unsqueeze(-1) + 1e-6)
        similarities = torch.einsum('bij,bkj->bik', gradient_1, gradient_2)
        return similarities.mean()
    else:
        similarities = torch.sign(torch.einsum('bij,bkj->bik', gradient_1, gradient_2))
        return similarities.mean()

def random_test_stiffness(B=4, N=10, M=12, S=768):
    gradient_1 = torch.randn(B, N, S)
    gradient_2 = torch.randn(B, M, S)

    cosine_stiffness = stiffness(gradient_1, gradient_2, s_type="cosine")
    sign_stiffness = stiffness(gradient_1, gradient_2, s_type="sign")

    assert len(cosine_stiffness.shape) == 0 and len(sign_stiffness.shape) == 0
    assert -1 <= cosine_stiffness <= 1, -1 <= sign_stiffness <= 1

random_test_stiffness(B=4, N=10, M=12, S=768)