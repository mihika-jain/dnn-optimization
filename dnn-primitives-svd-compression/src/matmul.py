import torch
import torch.nn.functional as F
import numpy as np

def matmul(A, B, method='naive', **kwargs):
    """
    Multiply two matrices.
    :param A: (N, M) torch tensor.
    :param B: (M, K) torch tensor.
    :param method:
    :return:
        Output matrix with shape (N, K)
    """
    method = method.lower()
    if method in ['naive', 'pytorch', 'torch']:
        return naive(A, B)
    elif method == 'svd':
        return svd(A, B, **kwargs)
    elif method in ['log', 'logmatmul']:
        return logmatmul(A, B, **kwargs)
    else:
        raise ValueError("Invalid [method] value: %s" % method)

def naive(A, B, **kwargs):
    return A @ B

def svd(A, B, rank_A=None, rank_B=None):
    """
    Apply low-rank approximation (SVD) to both matrix A and B with rank rank_A
    and rank_B respectively.
    :param A: (N, M) pytorch tensor
    :param B: (M, K) pytorch tensor
    :param rank_A: None or int. None means use original A matrix.
    :param rank_B: None or int. None means use original B matrix.
    :return: a (N, K) pytorch tensor
    """
    if torch.isnan(A).any() or torch.isinf(A).any():
        A = torch.nan_to_num(A, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if torch.isnan(B).any() or torch.isinf(B).any():
        B = torch.nan_to_num(B, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if rank_A is not None:
        try:
            U_A, S_A, V_A = torch.svd(A)
            rank_A = min(rank_A, min(A.shape))
            A_approx = U_A[:, :rank_A] @ torch.diag(S_A[:rank_A]) @ V_A[:, :rank_A].t()
        except RuntimeError:
            A_approx = A
    else:
        A_approx = A
        
    if rank_B is not None:
        try:
            U_B, S_B, V_B = torch.svd(B)
            rank_B = min(rank_B, min(B.shape))
            B_approx = U_B[:, :rank_B] @ torch.diag(S_B[:rank_B]) @ V_B[:, :rank_B].t()
        except RuntimeError:
            B_approx = B
    else:
        B_approx = B
        
    return A_approx @ B_approx


def logmatmul(A, B, **kwargs):
    """ TODO: use log multiplication for matrix-matrix multiplication """
    sign_A = torch.sign(A)
    sign_B = torch.sign(B)
    log_A = torch.log(torch.abs(A))
    log_B = torch.log(torch.abs(B))

    log_A_3d = log_A[:, :, None]
    log_B_3d = log_B[None, :, :]
    sign_A_3d = sign_A[:, :, None]
    sign_B_3d = sign_B[None, :, :]
  
    combined_log = log_A_3d + log_B_3d
    combined_sign = sign_A_3d * sign_B_3d

    combined_exp = torch.exp(combined_log)
    result = combined_sign * combined_exp

    final_output = result.sum(dim=1)

    return final_output
