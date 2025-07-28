import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from functools import partial
from collections.abc import Callable
from generative.networks.nets import SPADEDiffusionModelUNet
from generative.utils import unsqueeze_right

from monai.utils import optional_import

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")

def sample(
        self,
        input_noise: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor] | None = None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        verbose: bool = True,
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired sample.
            diffusion_model: model to sample from.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            verbose: if true, prints the progression bar of the sampling process.
            seg: if diffusion model is instance of SPADEDiffusionModel, segmentation must be provided.
        """
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")

        if not scheduler:
            print("No scheduler found")
            scheduler = self.scheduler
        image = input_noise
        if verbose and has_tqdm:
            print("IS verbose AND has tqdm")
            progress_bar = tqdm(scheduler.timesteps)
        else:
            print("IS NOT verbose OR has NO tqdm")
            progress_bar = iter(scheduler.timesteps)
        intermediates = []
        for t in progress_bar:
            # 1. predict noise model_output
            diffusion_model = (
                partial(diffusion_model, seg=seg)
                if isinstance(diffusion_model, SPADEDiffusionModelUNet)
                else diffusion_model
            )
            if mode == "concat":
                model_input = torch.cat([image, conditioning], dim=1)
                model_output = diffusion_model(
                    model_input, timesteps=torch.Tensor((t,)).to(input_noise.device), context=None
                )
            else:
                model_output = diffusion_model(
                    image, timesteps=torch.Tensor((t,)).to(input_noise.device), context=conditioning
                )

            # 2. compute previous image: x_t -> x_t-1
            image, _ = scheduler.step(model_output, t, image)
            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(image)
        if save_intermediates:
            return image, intermediates
        else:
            return image

def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Add noise to the original samples.

        Args:
            original_samples: original samples
            noise: noise to add to samples
            timesteps: timesteps tensor indicating the timestep to be computed for each sample.

        Returns:
            noisy_samples: sample with added noise
        """
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_cumprod = unsqueeze_right(self.alphas_cumprod[timesteps] ** 0.5, original_samples.ndim)
        sqrt_one_minus_alpha_prod = unsqueeze_right((1 - self.alphas_cumprod[timesteps]) ** 0.5, original_samples.ndim)

        noisy_samples = sqrt_alpha_cumprod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples