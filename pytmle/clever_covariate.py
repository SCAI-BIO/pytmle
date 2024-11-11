import numpy as np

def get_clever_covariate(g_star, nuisance_weight, h_fs, leq_j):
    """
    Calculates the clever covariate by adjusting nuisance weights and combining with hFS and LeqJ.

    Args:
        g_star (np.ndarray): Array representing the intervention regime.
        nuisance_weight (np.ndarray): Matrix of nuisance weights.
        h_fs (np.ndarray): Matrix representing hFS values.
        leq_j (int): Integer to compare against hFS.

    Returns:
        np.ndarray: Adjusted clever covariate matrix.
    """
    for i in range(nuisance_weight.shape[1]):
        nuisance_weight[:, i] *= g_star[i]
    return nuisance_weight * (leq_j - h_fs)


def get_haz_ls(t_tilde, eval_times, haz_l):
    """
    Adjusts each column of HazL based on the condition EvalTimes <= T_Tilde for each individual.

    Args:
        t_tilde (np.ndarray): Array of observed times.
        eval_times (np.ndarray): Array of evaluation times.
        haz_l (np.ndarray): Matrix of hazard values for each evaluation time and individual.

    Returns:
        np.ndarray: Adjusted hazard matrix.
    """
    for i in range(haz_l.shape[1]):
        haz_l[:, i] = np.where(eval_times <= t_tilde[i], haz_l[:, i], 0)
    return haz_l
