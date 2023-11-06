import torch
import torch.nn as nn

def diversity_matrix(expert_output):
    """
    Calculate the diversity matrix between experts' output.

    This function computes the pairwise diversity matrix between experts' output based on the Euclidean metric.

    Args:
        expert_output (torch.Tensor): A tensor of expert outputs where each column represents the output of one expert.

    Returns:
        torch.Tensor: A diversity matrix where each element at position (i, j) represents the diversity between experts i and j standardized to be between 0 and 1.

    """
    n_experts = len(expert_output.T)
    metric = nn.PairwiseDistance()
    diversity = torch.zeros([n_experts,n_experts])
    for i in range(n_experts):
        for j in range(n_experts):
            if i>j:
                diversity[i,j] = metric(expert_output[:,i],expert_output[:,j])
    return (diversity+diversity.T)/torch.max(diversity)

def permanent(diversity_matrix):
    """
    Calculate the permanent of a square matrix.

    The permanent of a square matrix is a value that is computed as the sum of all possible products of entries
    from each row such that each column is used exactly once. This function recursively computes the permanent of
    a given square matrix.

    Args:
        diversity_matrix (torch.Tensor): The square matrix for which the permanent is to be calculated.

    Returns:
        int: The permanent of the given square matrix.

    """
    def per(mtx, column, selected, prod):
        """
        Calculate the permanent recursively for a square matrix.

        This recursive function is used to calculate the permanent of a square matrix. The permanent is computed as the sum
        of all possible products of entries from each row such that each column is used exactly once.

        Args:
            mtx (torch.Tensor): The square matrix for which the permanent is being calculated.
            column (int): The current column being considered.
            selected (list of int): The list of selected rows for each column.
            prod (int): The product of the selected entries so far.

        Returns:
            int: The calculated permanent value for the given square matrix.

        """
        if column == mtx.shape[1]:
            return prod
        else:
            result = 0
            for row in range(mtx.shape[0]):
                result = result + per(mtx, column+1, selected+[row], prod*mtx[row,column])
            return result
    
    return per(diversity_matrix, 0, [], 1)
