from gaussian_model import GaussianModel

class RigidModel(GaussianModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._point_ids = torch.empty(0,device = 'cuda')  # (N, 1)
        self.instance_pose = torch.empty(0,device = 'cuda')  # (Frame, instancce, 4, 4)
