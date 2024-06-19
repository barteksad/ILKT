import torch

def stiffness(gradient_1, gradient_2, s_type="cosine"):
    g_1 = gradient_1.reshape(-1)
    g_2 = gradient_2.reshape(-1)
    if s_type == "cosine":
        g_1 = g_1/torch.norm(g_1, p=2)
        g_2 = g_2/torch.norm(g_2, p=2)
        return torch.dot(g_1, g_2)
    else:
        return torch.sign(torch.dot(g_1, g_2))

def random_test_stiffness(N=10, M=15):
    gradient_1 = torch.randn(N, M)
    gradient_2 = torch.randn_like(gradient_1)

    cosine_stiffness = stiffness(gradient_1, gradient_2)
    sign_stiffness = stiffness(gradient_1, gradient_2, s_type="sign")

    assert len(cosine_stiffness.shape) == 0 and len(sign_stiffness.shape) == 0
    assert -1 <= cosine_stiffness <= 1, -1 <= sign_stiffness <= 1
