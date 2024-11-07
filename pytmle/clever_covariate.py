import numpy as np

def getCleverCovariate(GStar, NuisanceWeight, hFS, LeqJ):
    """
    Calculate the clever covariate for Influence Curve calculation.

    Parameters:
    GStar (np.array): Vector of modified propensity scores.
    NuisanceWeight (np.array): Matrix of nuisance weights.
    hFS (np.array): Contribution of cumulative incidence function to clever covariate.
    LeqJ (int): Indicator for event condition.

    Returns:
    np.array: Matrix representing the clever covariate.
    """
    # Scale each column of NuisanceWeight by the corresponding GStar value
    NuisanceWeight = NuisanceWeight * GStar[:, np.newaxis]
    
    # Return the element-wise product with (LeqJ - hFS)
    return NuisanceWeight * (LeqJ - hFS)


def getHazLS(T_Tilde, EvalTimes, HazL):
    """
    Adjust hazard matrix based on observed event times.

    Parameters:
    T_Tilde (np.array): Observed event times.
    EvalTimes (np.array): Evaluation times.
    HazL (np.array): Hazard matrix.

    Returns:
    np.array: Adjusted hazard matrix.
    """
    for i in range(HazL.shape[1]):
        # Create a mask where EvalTimes <= T_Tilde[i]
        mask = EvalTimes <= T_Tilde[i]
        HazL[:, i] = HazL[:, i] * mask  # Apply the mask to each column
    return HazL
