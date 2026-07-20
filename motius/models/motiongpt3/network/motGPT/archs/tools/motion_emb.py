from torch import nn

class MotionUndHead(nn.Module):
    def __init__(self, input_dim, output_dim, projector_type='linear', depth=1, **kwargs):
        super().__init__()
        if projector_type == "identity":
            modules = nn.Identity()

        elif projector_type == "linear":
            modules = nn.Linear(input_dim, output_dim)

        elif projector_type == "mlp_gelu":
            mlp_depth = depth
            modules = [nn.Linear(input_dim, output_dim)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(output_dim, output_dim))
            modules = nn.Sequential(*modules)
        
        else:
            raise ValueError(f"Unknown projector type: {projector_type}")
        
        self.layers = modules

    def forward(self, motion_tokens):
        if len(motion_tokens.shape) == 3:
            bs, latent_dim, latent_channel = motion_tokens.shape
            motion_tokens = motion_tokens.permute(1,0,2)
            motion_tokens = motion_tokens.reshape(bs, -1)
        motion_embedding = self.layers(motion_tokens)
        return motion_embedding

