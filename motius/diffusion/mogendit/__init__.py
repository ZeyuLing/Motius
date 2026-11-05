from motius.diffusion.mogendit.base_diffusion import GaussianDiffusion
from motius.diffusion.mogendit.base_diffusion import BetaSchedule
from motius.diffusion.mogendit.base_diffusion import ModelMeanType



def create_diffusion():
    return GaussianDiffusion(num_timesteps=1000, beta_schedule=BetaSchedule.COSINE,
                             model_mean_type=ModelMeanType.START_X)
