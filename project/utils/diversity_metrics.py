import torch
import torch.nn as nn

def diversity_matrix(expert_output):
    n_experts = len(expert_output.T)
    metric = nn.PairwiseDistance()
    diversity = torch.zeros([n_experts,n_experts])
    for i in range(n_experts):
        for j in range(n_experts):
            if i>j:
                diversity[i,j] = metric(expert_output[:,i],expert_output[:,j])
    return (diversity+diversity.T)/torch.max(diversity)

def permanent(diversity_matrix):
    def per(mtx, column, selected, prod):
        """
        Row expansion for the permanent of matrix mtx.
        The counter column is the current column, 
        selected is a list of indices of selected rows,
        and prod accumulates the current product.
        """
        if column == mtx.shape[1]:
            return prod
        else:
            result = 0
            for row in range(mtx.shape[0]):
                result = result + per(mtx, column+1, selected+[row], prod*mtx[row,column])
            return result
    
    return per(diversity_matrix, 0, [], 1)
