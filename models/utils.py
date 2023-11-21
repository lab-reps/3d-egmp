import torch

def coord2radial(edge_index, coord, epsilon=1e-8):
    row, col=edge_index
    radial=torch.sum((coord[row]-coord[col])**2, 1).unsqueeze(1)

    if self.normalize:
        norm=torch.sqrt(radial).detach()+epsilon
        coord_diff=coord_diff/norm
    
    return radial, coord_diff