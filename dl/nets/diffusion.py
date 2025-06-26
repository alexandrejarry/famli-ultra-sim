import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import functools

from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks import nets
from generative.networks.nets import PatchDiscriminator
from generative.losses import PatchAdversarialLoss, PerceptualLoss


from diffusers import DDPMScheduler, UNet2DModel

import lightning as L
from lightning.pytorch.core import LightningModule
from transforms import ultrasound_transforms as ust 

from typing import Union

class AutoEncoderKL(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        latent_channels = 3
        if hasattr(self.hparams, "latent_channels"):
            latent_channels = self.hparams.latent_channels

        self.autoencoderkl = nets.AutoencoderKL(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(128, 256, 384),
            latent_channels=latent_channels,
            num_res_blocks=1,
            norm_num_groups=32,
            attention_levels=(False, False, True),
        )

        # self.autoencoderkl = nets.AutoencoderKL(
        #     spatial_dims=2,
        #     in_channels=1,
        #     out_channels=1,
        #     num_channels=(128, 128, 256, 512),
        #     latent_channels=latent_channels,
        #     num_res_blocks=2,
        #     attention_levels=(False, False, False, False),
        #     with_encoder_nonlocal_attn=False,
        #     with_decoder_nonlocal_attn=False,
        # )

        # self.autoencoderkl = nets.AutoencoderKL(spatial_dims=2,
        #     in_channels=1,
        #     out_channels=1,
        #     num_channels=(128, 256, 512, 512),
        #     latent_channels=latent_channels,
        #     num_res_blocks=2,
        #     norm_num_groups=32,
        #     attention_levels=(False, False, False, True))

        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")

        self.automatic_optimization = False

        self.discriminator = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)        

        self.adversarial_loss = PatchAdversarialLoss(criterion="least_squares")        

        # For mixed precision training
        # self.scaler_g = GradScaler()
        # self.scaler_d = GradScaler()

        if hasattr(self.hparams, "denoise") and self.hparams.denoise: 
            self.noise_transform = torch.nn.Sequential(
                ust.GaussianNoise(0.0, 0.05)
            )
        else:
            self.noise_transform = nn.Identity()
        

    def configure_optimizers(self):
        optimizer_g = optim.AdamW(self.autoencoderkl.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        optimizer_d = optim.AdamW(self.discriminator.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        return [optimizer_g, optimizer_d]
    
    def compute_loss_generator(self, x, reconstruction, z_mu, z_sigma):
        recons_loss = F.l1_loss(reconstruction.float(), x.float())
        p_loss = self.perceptual_loss(reconstruction.float(), x.float())
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss_g = recons_loss + (self.hparams.kl_weight * kl_loss) + (self.hparams.perceptual_weight * p_loss)

        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:
            logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
            generator_adv_loss = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g += self.hparams.adversarial_weight * generator_adv_loss

        return loss_g, recons_loss
    
    def compute_loss_discriminator(self, x, reconstruction):
        logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
        loss_d_fake = self.adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = self.discriminator(x.contiguous().detach())[-1]
        loss_d_real = self.adversarial_loss(logits_real, target_is_real=True, for_discriminator=True)
        return (loss_d_fake + loss_d_real) * 0.5


    def training_step(self, train_batch, batch_idx):
        x = train_batch

        optimizer_g, optimizer_d = self.optimizers()

        reconstruction, z_mu, z_sigma = self.autoencoderkl(self.smooth_transform(self.noise_transform(x)))

        loss_g, recons_loss = self.compute_loss_generator(x, reconstruction, z_mu, z_sigma)

        optimizer_g.zero_grad()
        self.manual_backward(loss_g)
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)
        
        loss_d = 0.0
        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:
            loss_d = self.compute_loss_discriminator(x, reconstruction)

            optimizer_d.zero_grad()
            self.manual_backward(loss_d)
            optimizer_d.step()
            self.untoggle_optimizer(optimizer_d)

        self.log("train_loss_recons", recons_loss)
        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)

        return {"train_loss_g": loss_g, "train_loss_d": loss_d}
        

    def validation_step(self, val_batch, batch_idx):
        x = val_batch

        reconstruction, z_mu, z_sigma = self.autoencoderkl(x)
        recon_loss = F.l1_loss(x.float(), reconstruction.float())

        self.log("val_loss", recon_loss, sync_dist=True)

    def forward(self, images):        
        return self.autoencoderkl(images)



class DiffusionModel(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.net = UNet2DModel(
            in_channels=self.hparams.in_channels,  # the number of input channels, 3 for RGB images
            out_channels=self.hparams.out_channels,  # the number of output channels
            layers_per_block=self.hparams.layers_per_block,  # how many ResNet layers to use per UNet block
            block_out_channels=self.hparams.block_out_channels,  # the number of output channes for each UNet block
            down_block_types=self.hparams.down_block_types,  # the types of blocks to use in the downsampling part of the UNet
            up_block_types=self.hparams.up_block_types,
        )
        self.scheduler = DDPMScheduler(num_train_timesteps=self.hparams.num_train_timesteps, beta_schedule="linear")
        self.loss_fn = nn.MSELoss()

        self.resize = ust.Resize2D(self.hparams.image_size)

    @staticmethod
    def add_model_specific_args(parent_parser):

        hparams_group = parent_parser.add_argument_group('DiffusionModel Hugging Face')
        hparams_group.add_argument('--lr', default=1e-4, type=float, help='Learning rate generator')
        hparams_group.add_argument('--num_train_timesteps', default=1000, type=int, help='Number of training timesteps')
        hparams_group.add_argument('--in_channels', default=1, type=int, help='Number of input channels')
        hparams_group.add_argument('--out_channels', default=1, type=int, help='Number of output channels')
        hparams_group.add_argument('--layers_per_block', default=2, type=int, help='Number of layers per block')
        hparams_group.add_argument('--block_out_channels', default=(128, 128, 256, 256, 512, 512), nargs='+', type=int, help='Block out channels')
        hparams_group.add_argument('--down_block_types', default=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"), nargs='+', type=str, help='Down block types')
        hparams_group.add_argument('--up_block_types', default=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"), nargs='+', type=str, help='Up block types')
        hparams_group.add_argument('--image_size', default=(128, 128), type=int, nargs='+', help='Image size')

        return parent_parser
        

    def forward(self, x, timesteps):
        return self.net(x, timesteps).sample

    def training_step(self, batch, batch_idx):
        x = batch  

        x = self.resize(x)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (x.size(0),), device=self.device).long()
        noise = torch.randn_like(x).to(self.device)
        noisy_images = self.scheduler.add_noise(x, noise, timesteps)

        pred = self(noisy_images, timesteps)  # Model forward pass
        loss = self.loss_fn(pred, noise)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch  
        x = self.resize(x)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (x.size(0),), device=self.device).long()
        noise = torch.randn_like(x).to(self.device)
        noisy_images = self.scheduler.add_noise(x, noise, timesteps)

        pred = self(noisy_images, timesteps)  # Model forward pass
        loss = self.loss_fn(pred, noise)

        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def sample(self, num_samples=1):

        size = list(self.hparams.image_size)
        size = [num_samples, self.hparams.in_channels] + size

        noisy_image = torch.randn(size).to(self.device)
        # Sample timesteps
        timesteps = self.scheduler.timesteps.to(self.device)

        # Reverse process (denoising)
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                # Predict noise
                noise_pred = self.net(noisy_image, t.unsqueeze(0)).sample
                
                # Remove noise using scheduler
                noisy_image = self.scheduler.step(noise_pred, t, noisy_image).prev_sample

        return noisy_image

class USDDPMPC(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.autoencoder_fn is not None:
            au = AutoEncoderKL.load_from_checkpoint(self.hparams.autoencoder_fn)
            au.freeze()
            au = au.autoencoderkl
        else:
            au = nets.AutoencoderKL(
                spatial_dims=2,
                in_channels=self.hparams.in_channels,
                out_channels=self.hparams.out_channels,
                num_channels=self.hparams.num_channels,
                latent_channels=self.hparams.latent_channels,
                num_res_blocks=self.hparams.num_res_blocks,
                norm_num_groups=self.hparams.norm_num_groups,
                attention_levels=(False, False, True),
            )
        self.encoder = TimeDistributed(au.encoder, time_dim=2)
        self.quant_conv_mu = TimeDistributed(au.quant_conv_mu, time_dim=2)
        self.quant_conv_log_sigma = TimeDistributed(au.quant_conv_log_sigma, time_dim=2)
        
        self.vs = VolumeSamplingBlindSweep()

        self.attn_chunk = AttentionChunk(input_dim=(self.hparams.latent_channels*64*64), hidden_dim=64, chunks=self.hparams.n_chunks)
        self.proj = ProjectionHead(input_dim=(self.hparams.latent_channels*64*64), hidden_dim=1280, output_dim=self.hparams.embed_dim)

        self.dropout = nn.Dropout(self.hparams.dropout)

        self.p_encoding = PositionalEncoding2D(self.hparams.embed_dim)

        self.diffnet = UNet2DConditionModel(
            in_channels=self.hparams.input_dim, 
            out_channels=self.hparams.output_dim, 
            cross_attention_dim=self.hparams.embed_dim, 
            flip_sin_to_cos=self.hparams.flip_sin_to_cos,
            time_embedding_type=self.hparams.time_embedding_type,
            freq_shift=self.hparams.freq_shift)
        

        self.loss_fn = nn.MSELoss()

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.hparams.num_train_steps, 
                                             beta_schedule="linear",
                                             clip_sample=False,
                                             prediction_type='epsilon')

        self.simu0 = TimeDistributed(us_simulation_jit.MergedLinearCutLabel11(), time_dim=2)
        self.simu1 = TimeDistributed(us_simulation_jit.MergedCutLabel11(), time_dim=2)
        self.simu2 = TimeDistributed(us_simulation_jit.MergedUSRLabel11(), time_dim=2)
        self.simu3 = TimeDistributed(us_simulation_jit.MergedLinearLabel11(), time_dim=2)
        self.simu4 = TimeDistributed(us_simulation_jit.MergedLinearLabel11WOG(), time_dim=2)

        
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("USDDPMPC")

        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
        # Image Encoder parameters 

        group.add_argument("--autoencoder_fn", type=str, default="/mnt/raid/C1_ML_Analysis/train_output/diffusionAE/extract_frames_Dataset_C_masked_resampled_256_spc075_wscores_meta_BPD01_MACFL025-7mo-9mo/v0.4/epoch=72-val_loss=0.01.ckpt", help='Pre trained autoencoder model')
        group.add_argument("--latent_channels", type=int, default=3, help='Output dimension for the image encoder')
        group.add_argument("--spatial_dims", type=int, default=2, help='Spatial dimensions for the autoencoder')
        group.add_argument("--in_channels", type=int, default=1, help='Input channels for the autoencoder')
        group.add_argument("--out_channels", type=int, default=1, help='Output channels for the autoencoder')
        group.add_argument("--num_channels", type=int, nargs="*", default=[128, 256, 384], help='Number of channels for each stage of the image encoder')
        group.add_argument("--num_res_blocks", type=int, default=1, help='Number of residual blocks')
        group.add_argument("--norm_num_groups", type=int, default=32, help='Number of groups for the normalization layer')
        group.add_argument("--n_chunks_e", type=int, default=10, help='Number of chunks in the encoder stage to reduce memory usage')
        group.add_argument("--n_chunks", type=int, default=8, help='Number of outputs in the time dimension')
        group.add_argument("--z_prob", type=float, default=0.2, help='Probability of dropping the latent code')
        
        
        # Encoder parameters for the diffusion model
        group.add_argument("--num_samples", type=int, default=4096, help='Number of samples to take from the mesh to start the encoding')
        group.add_argument("--input_dim", type=int, default=3, help='Input dimension for the encoder')
        group.add_argument("--embed_dim", type=int, default=1280, help='Embedding dimension')
        group.add_argument("--output_dim", type=int, default=3, help='Output dimension of the model')
        group.add_argument("--dropout", type=float, default=0.25, help='Dropout rate')
        
        group.add_argument("--num_train_steps", type=int, default=1000, help='Number of training steps for the noise scheduler')
        group.add_argument("--time_embedding_type", type=str, default='positional', help='Time embedding type', choices=['fourier', 'positional'])
        
        group.add_argument("--flip_sin_to_cos", type=int, default=1, help='Whether to flip sin to cos for Fourier time embedding.')
        group.add_argument("--freq_shift", type=int, default=0, help='Frequency shift for Fourier time embedding.')

        group.add_argument("--num_random_sweeps", type=int, default=0, help='How many random sweeps to use. 0 == all')
        group.add_argument("--n_grids", type=int, default=200, help='Number of grids')
        # group.add_argument("--target_label", type=int, default=7, help='Target label')

        # group.add_argument("--n_fixed_samples", type=int, default=12288, help='Number of fixed samples')

        return parent_parser
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer

    def on_fit_start(self):        

        # Define the file names directly without using out_dir
        grid_t_file = 'grid_t.pt'
        inverse_grid_t_file = 'inverse_grid_t.pt'
        mask_fan_t_file = 'mask_fan_t.pt'

        if not os.path.exists(grid_t_file):
            grid_tensor = []
            inverse_grid_t = []
            mask_fan_t = []

            for i in range(self.hparams.n_grids):

                grid_w, grid_h = self.hparams.grid_w, self.hparams.grid_h
                center_x = self.hparams.center_x
                r1 = self.hparams.r1

                center_y = self.hparams.center_y_start + (self.hparams.center_y_end - self.hparams.center_y_start) * (torch.rand(1))
                r2 = self.hparams.r2_start + ((self.hparams.r2_end - self.hparams.r2_start) * torch.rand(1)).item()
                theta = self.hparams.theta_start + ((self.hparams.theta_end - self.hparams.theta_start) * torch.rand(1)).item()
                
                grid, inverse_grid, mask = self.USR.init_grids(grid_w, grid_h, center_x, center_y, r1, r2, theta)

                grid_tensor.append(grid.unsqueeze(dim=0))
                inverse_grid_t.append(inverse_grid.unsqueeze(dim=0))
                mask_fan_t.append(mask.unsqueeze(dim=0))

            self.grid_t = torch.cat(grid_tensor).to(self.device)
            self.inverse_grid_t = torch.cat(inverse_grid_t).to(self.device)
            self.mask_fan_t = torch.cat(mask_fan_t).to(self.device)

            # Save tensors directly to the current directory
            
            torch.save(self.grid_t, grid_t_file)
            torch.save(self.inverse_grid_t, inverse_grid_t_file)
            torch.save(self.mask_fan_t, mask_fan_t_file)

            # print("Grids SAVED!")
            # print(self.grid_t.shape, self.inverse_grid_t.shape, self.mask_fan_t.shape)
        
        elif not hasattr(self, 'grid_t'):
            # Load tensors directly from the current directory
            self.grid_t = torch.load(grid_t_file).to(self.device)
            self.inverse_grid_t = torch.load(inverse_grid_t_file).to(self.device)
            self.mask_fan_t = torch.load(mask_fan_t_file).to(self.device)

    def sampling(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:        
        eps = torch.randn_like(z_sigma)
        z_vae = z_mu + eps * z_sigma
        return z_vae
    
    def compute_loss(self, X, X_hat, step="train", sync_dist=False):

        # Negative ELBO of P(X|z)
        loss = self.loss_fn(X, X_hat)
        
        self.log(f"{step}_loss", loss, sync_dist=sync_dist)

        return loss

    def volume_sampling(self, X, X_origin, X_end, use_random=False):
        with torch.no_grad():
            simulator = self.simu0
            
            grid = None
            inverse_grid = None
            mask_fan = None

            tags = self.vs.tags

            if use_random:

                simulator_idx = np.random.choice([0, 1, 2, 3, 4], replace=False)
                simulator = getattr(self, f'simu{simulator_idx}')

                if self.hparams.num_random_sweeps > 0:
                    tags = np.random.choice(self.vs.tags, self.hparams.num_random_sweeps)
                else:
                    tags = self.vs.tags

                grid_idx = torch.randint(low=0, high=self.hparams.n_grids - 1, size=(1,))
            
                grid = self.grid_t[grid_idx]
                inverse_grid = self.inverse_grid_t[grid_idx]
                mask_fan = self.mask_fan_t[grid_idx] 

            X_sweeps = []
            X_sweeps_tags = []            

            for tag in tags:                
                
                sampled_sweep_simu = self.vs.get_sweep(X, X_origin, X_end, tag, use_random=use_random, simulator=simulator, grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan)
                X_sweeps.append(sampled_sweep_simu)
                X_sweeps_tags.append(self.vs.tags_dict[tag])
                
            X_sweeps = torch.cat(X_sweeps, dim=1)
            X_sweeps_tags = torch.tensor(X_sweeps_tags).repeat(X_sweeps.shape[0], 1)

            return X_sweeps, X_sweeps_tags

    def positional_encoding(self, z, sweeps_tags):
        # Positional encoding for training only. Ensure the encoding is done in the correct order depending on the sweep
        # Define shapes
        BS, N, C, F = z.shape # [BS, N, self.hparams.n_chunks, 64*64*self.hparams.latent_channels]
        max_N = len(self.vs.tags)
        
        buffer = torch.zeros((BS, max_N, C, F), dtype=torch.float32, device=self.device)  # Buffer for data

        # Scatter data into the buffer
        for b in range(BS):
            buffer[b, sweeps_tags[b]] = z[b]

        buffer = self.p_encoding(buffer)
        
        # Assign output in tensor
        for b in range(BS):
            z[b] = buffer[b, sweeps_tags[b]]

        return z

    def training_step(self, train_batch, batch_idx):
        X, X_origin, X_end, X_PC = train_batch

        X, rotation_matrices = self.vs.random_rotate_3d_batch(X)

        batch_size = X.shape[0]

        if  torch.rand(1).item() < self.hparams.z_prob:
            z = torch.zeros(batch_size, 1, self.hparams.embed_dim, device=self.device)
        else:
            

            x_sweeps, sweeps_tags = self.volume_sampling(X, X_origin, X_end, use_random=True)

            # x_sweeps shape is B, N, C, T, H, W. N for number of sweeps ex. torch.Size([2, 2, 1, 200, 256, 256])
            # tags shape torch.Size([2, 2])
            Nsweeps = x_sweeps.shape[1] # Number of sweeps -> T
            
            z = []
            x_v = []

            for n in range(Nsweeps):
                x_sweeps_n = x_sweeps[:, n, :, :, :, :] # [BS, C, T, H, W]
                sweeps_tags_n = sweeps_tags[:, n]
                
                z_mu, z_sigma = self.encode(x_sweeps_n)            
                z_ = self.sampling(z_mu, z_sigma) 

                z_ = self.attn_chunk(z_) # [BS, self.hparams.latent_channels, self.hparams.n_chunks, 64. 64]

                z_ = z_.permute(0, 2, 3, 4, 1).reshape(batch_size, self.hparams.n_chunks, -1) # [BS, self.hparams.n_chunks, 64*64*self.hparams.latent_channels]

                z.append(z_.unsqueeze(1))

            z = torch.cat(z, dim=1) # [BS, N, self.hparams.n_chunks, 64*64*self.hparams.latent_channels]

            z = self.proj(z) # [BS, N, self.hparams.n_chunks, 1280]

            z = self.positional_encoding(z, sweeps_tags)
            z = z.view(batch_size, -1, self.hparams.embed_dim).contiguous()
            z = self.dropout(z)

        # Diffusion stage

        X_PC = self.vs.apply_batch_rotation(X_PC, rotation_matrices[:,0:3,0:3])

        noise = torch.randn_like(X_PC).to(self.device)

        timesteps = torch.randint(0, self.hparams.num_train_steps - 1, (X_PC.shape[0],)).long().to(self.device)

        noisy_X = self.noise_scheduler.add_noise(X_PC, noise=noise, timesteps=timesteps)

        X_hat = self(noisy_X.permute(0, 2, 1).view(-1, 3, 64, 64).contiguous(), timesteps=timesteps, context=z)
        X_hat = X_hat.view(-1, 3, 64*64).permute(0, 2, 1).contiguous()

        return self.compute_loss(X=noise, X_hat=X_hat)
        

    def validation_step(self, val_batch, batch_idx):
        
        X, X_origin, X_end, X_PC = val_batch

        x_sweeps, sweeps_tags = self.volume_sampling(X, X_origin, X_end)

        # x_sweeps shape is B, N, C, T, H, W. N for number of sweeps ex. torch.Size([2, 2, 1, 200, 256, 256]) 
        # tags shape torch.Size([2, 2])

        batch_size = x_sweeps.shape[0]
        Nsweeps = x_sweeps.shape[1] # Number of sweeps -> T
        
        z = []
        x_v = []

        for n in range(Nsweeps):
            x_sweeps_n = x_sweeps[:, n, :, :, :, :] # [BS, C, T, H, W]
            sweeps_tags_n = sweeps_tags[:, n]

            z_mu, z_sigma = self.encode(x_sweeps_n)
            z_ = z_mu

            z_ = self.attn_chunk(z_) # [BS, self.hparams.latent_channels, self.hparams.n_chunks, 64. 64]

            z_ = z_.permute(0, 2, 3, 4, 1).reshape(batch_size, self.hparams.n_chunks, -1) # [BS, self.hparams.n_chunks, 64*64*self.hparams.latent_channels]

            z.append(z_.unsqueeze(1))

        z = torch.cat(z, dim=1) # [BS, N, self.hparams.n_chunks, 64*64*self.hparams.latent_channels]

        z = self.proj(z) # [BS, N, elf.hparams.n_chunks, 1280]

        # We don't need to do the trick of using the buffer for the positional encoding here, ALL the sweeps are present in validation
        z = self.p_encoding(z)
        z = z.view(batch_size, -1, self.hparams.embed_dim).contiguous()

        # Diffusion stage
        noise = torch.randn_like(X_PC).to(self.device)

        timesteps = torch.randint(0, self.hparams.num_train_steps - 1, (X_PC.shape[0],)).long().to(self.device)

        noisy_X = self.noise_scheduler.add_noise(X_PC, noise=noise, timesteps=timesteps)

        X_hat = self(noisy_X.permute(0, 2, 1).view(-1, 3, 64, 64).contiguous(), timesteps=timesteps, context=z)
        X_hat = X_hat.view(-1, 3, 64*64).permute(0, 2, 1).contiguous()

        return self.compute_loss(X=noise, X_hat=X_hat, step="val", sync_dist=True)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        """
        Forwards an image through the spatial encoder, obtaining the latent mean and sigma representations.

        Args:
            x: BxCx[SPATIAL DIMS] tensor

        """
        h = []
        for x_chunk in x.chunk(self.hparams.n_chunks_e, dim=2):
            h.append(self.encoder(x_chunk))
        h = torch.cat(h, dim=2)

        z_mu = self.quant_conv_mu(h)
        z_log_var = self.quant_conv_log_sigma(h)
        z_log_var = torch.clamp(z_log_var, -30.0, 20.0)
        z_sigma = torch.exp(z_log_var / 2)

        return z_mu, z_sigma

    def forward(self, x: torch.tensor, timesteps: Union[torch.Tensor, float, int], context: torch.Tensor | None = None):

        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=self.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(self.device)

        if timesteps.shape[0] != x.shape[0]:
            timesteps = timesteps.repeat(x.shape[0])
            
        return self.diffnet(sample=x, timestep=timesteps, encoder_hidden_states=context).sample

    def sample(self, num_samples=1, intermediate_steps=None, z=None):
        intermediates = []

        X_t = torch.randn(num_samples, self.hparams.num_samples, self.hparams.input_dim).to(self.device)        

        for i, t in enumerate(self.noise_scheduler.timesteps):

            
            x_hat = self(X_t.permute(0, 2, 1).view(-1, self.hparams.input_dim, 64, 64).contiguous(), timesteps=t, context=z)  
            x_hat = x_hat.view(-1, self.hparams.input_dim, 64*64).permute(0, 2, 1).contiguous()

            # Update sample with step
            X_t = self.noise_scheduler.step(model_output=x_hat, timestep=t, sample=X_t).prev_sample
            # X_t = X_t.clone().detach()
            if intermediate_steps is not None and intermediate_steps > 0 and t % (self.hparams.num_train_steps//intermediate_steps) == 0:
                intermediates.append(X_t)

        return X_t, intermediates

    def sample_guided(self, num_samples=1, guidance_scale=7.5, intermediate_steps=None, z=None):
        intermediates = []

        # Initialize random noise
        device = self.device
        x_t = torch.randn(num_samples, 64*64, self.hparams.input_dim, device=device)

        for t in reversed(range(self.hparams.num_train_steps)):
            
            # Conditional prediction (with context)
            x_cond = self(
                x_t.permute(0, 2, 1).view(-1, self.hparams.input_dim, 64, 64).contiguous(),
                timesteps=t,
                context=z
            )
            x_cond = x_cond.view(-1, self.hparams.input_dim, 64*64).permute(0, 2, 1)

            # Unconditional prediction (without context)
            x_uncond = self(
                x_t.permute(0, 2, 1).view(-1, self.hparams.input_dim, 64, 64).contiguous(),
                timesteps=t,
                context=torch.zeros(num_samples, 1, self.hparams.embed_dim, device=device)
            )
            x_uncond = x_uncond.view(-1, self.hparams.input_dim, 64*64).permute(0, 2, 1).contiguous()

            # Perform classifier-free guidance
            x_guided = x_uncond + guidance_scale * (x_cond - x_uncond)

            # Update the diffusion step using guided output
            x_t = self.noise_scheduler.step(model_output=x_guided, t=t, sample=x_t)

            # Save intermediate steps if needed
            if intermediate_steps and t % (self.hparams.num_train_steps // intermediate_steps) == 0:
                intermediates.append(x_t)

        return x_t, intermediates

    # Given diffusion model: model(x, t, cond)
# x: noisy input at timestep t
# cond: conditional information, cond=None is unconditional

def guided_sampling(model, x_t, t, cond, guidance_scale):
    # Predict noise conditioned on cond
    eps_cond = model(x_t, t, cond)
    
    # Predict noise without conditioning (unconditional)
    eps_uncond = model(x_t, t, cond=None)
    
    # Perform classifier-free guidance
    eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    
    return eps_guided


class USBabyFrame(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.vs = VolumeSamplingBlindSweep()
        
        encoder = monai.networks.nets.EfficientNetBN('efficientnet-b0', spatial_dims=2, in_channels=self.hparams.in_channels, num_classes=self.hparams.features)
        self.encoder = TimeDistributed(encoder, time_dim=2)

        p_encoding = torch.stack([self.positional_encoding(self.hparams.time_steps, self.hparams.features, tag) for tag in self.vs.tags_dict.values()])        
        self.register_buffer("p_encoding", p_encoding)
        
        self.proj = ProjectionHead(input_dim=self.hparams.features, hidden_dim=self.hparams.features, output_dim=self.hparams.embed_dim, activation=nn.LeakyReLU)
        self.attn_chunk = AttentionChunk(input_dim=self.hparams.embed_dim, hidden_dim=64, chunks=self.hparams.n_chunks)

        self.dropout = nn.Dropout(self.hparams.dropout)
        
        # 
        # self.mha = nn.MultiheadAttention(embed_dim=self.hparams.embed_dim, num_heads=self.hparams.num_heads, dropout=self.hparams.dropout, bias=False, batch_first=True)
        self.mha = MHAContextModulated(embed_dim=self.hparams.embed_dim, num_heads=self.hparams.num_heads, output_dim=self.hparams.embed_dim, dropout=self.hparams.dropout)
        
        # MHAContextModulated(embed_dim=self.hparams.embed_dim, num_heads=self.hparams.num_heads, output_dim=self.hparams.embed_dim, dropout=self.hparams.dropout)
        self.attn = SelfAttention(input_dim=self.hparams.embed_dim, hidden_dim=64)
        self.proj_final = ProjectionHead(input_dim=self.hparams.embed_dim, hidden_dim=64, output_dim=9, activation=nn.Tanh)        

        self.simu0 = TimeDistributed(us_simulation_jit.MergedLinearCutLabel11(), time_dim=2)
        self.simu1 = TimeDistributed(us_simulation_jit.MergedCutLabel11(), time_dim=2)
        self.simu2 = TimeDistributed(us_simulation_jit.MergedUSRLabel11(), time_dim=2)
        self.simu3 = TimeDistributed(us_simulation_jit.MergedLinearLabel11(), time_dim=2)
        self.simu4 = TimeDistributed(us_simulation_jit.MergedLinearLabel11WOG(), time_dim=2)
        
        belly_idx = np.load(self.hparams.belly_idx)
        self.register_buffer("belly_idx", torch.tensor(belly_idx, dtype=torch.long))
        head_idx = np.load(self.hparams.head_idx)        
        self.register_buffer("head_idx", torch.tensor(head_idx, dtype=torch.long))
        side_idx = np.load(self.hparams.side_idx)
        self.register_buffer("side_idx", torch.tensor(side_idx, dtype=torch.long))

        self.loss_fn = nn.CosineSimilarity(dim=2)
        # self.loss_fn = nn.MSELoss()

        
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("Calculate baby frame of orientation")

        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
        # Image Encoder parameters 
        group.add_argument("--spatial_dims", type=int, default=2, help='Spatial dimensions for the encoder')
        group.add_argument("--in_channels", type=int, default=1, help='Input channels for encoder')
        group.add_argument("--features", type=int, default=1280, help='Number of output features for the encoder')        
        group.add_argument("--n_chunks_e", type=int, default=16, help='Number of chunks in the encoder stage to reduce memory usage')
        group.add_argument("--n_chunks", type=int, default=16, help='Number of outputs in the time dimension')
        group.add_argument("--num_heads", type=int, default=8, help='Number of heads for multi_head attention')

        # Encoder parameters for the diffusion model
        group.add_argument("--num_samples", type=int, default=4096, help='Number of samples to take from the mesh to start the encoding')
        group.add_argument("--input_dim", type=int, default=3, help='Input dimension for the encoder')
        group.add_argument("--embed_dim", type=int, default=128, help='Embedding dimension')
        group.add_argument("--output_dim", type=int, default=3, help='Output dimension of the model')
        group.add_argument("--dropout", type=float, default=0.25, help='Dropout rate')
        
        
        group.add_argument("--time_steps", type=int, default=96, help='Number of time steps in the sweep or sequence length')
        group.add_argument("--num_random_sweeps", type=int, default=3, help='How many random sweeps to use. 0 == all')
        group.add_argument("--n_grids", type=int, default=200, help='Number of grids')
        group.add_argument("--belly_idx", type=str, default='/mnt/raid/C1_ML_Analysis/simulated_data_export/fetus_rest_selected/belly_idx.npy', help='Indices for belly')
        group.add_argument("--head_idx", type=str, default='/mnt/raid/C1_ML_Analysis/simulated_data_export/fetus_rest_selected/head_idx.npy', help='Indices for head')
        group.add_argument("--side_idx", type=str, default='/mnt/raid/C1_ML_Analysis/simulated_data_export/fetus_rest_selected/side_idx.npy', help='Indices for side')
        
        
        # group.add_argument("--p_ids", type=int, nargs='+', default=[1470, 3369, 2043], help='Point ids to compute the orthogonal orientation frame')
        # group.add_argument("--target_label", type=int, default=7, help='Target label')

        # group.add_argument("--n_fixed_samples", type=int, default=12288, help='Number of fixed samples')

        return parent_parser
    
    def positional_encoding(self, seq_len: int, d_model: int, tag: int) -> torch.Tensor:
        """
        Sinusoidal positional encoding with tag-based offset.

        Args:
            seq_len (int): Sequence length.
            d_model (int): Embedding dimension.
            tag (int): Unique tag for the sequence.
            device (str): Device to store the tensor.

        Returns:
            torch.Tensor: Positional encoding (seq_len, d_model).
        """
        pe = torch.zeros(seq_len, d_model)
        
        # Offset positions by a tag-dependent amount to make each sequence encoding unique
        position = torch.arange(tag * seq_len, (tag + 1) * seq_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer

    def on_fit_start(self):        

        # Define the file names directly without using out_dir
        grid_t_file = 'grid_t.pt'
        inverse_grid_t_file = 'inverse_grid_t.pt'
        mask_fan_t_file = 'mask_fan_t.pt'

        if not os.path.exists(grid_t_file):
            grid_tensor = []
            inverse_grid_t = []
            mask_fan_t = []

            for i in range(self.hparams.n_grids):

                grid_w, grid_h = self.hparams.grid_w, self.hparams.grid_h
                center_x = self.hparams.center_x
                r1 = self.hparams.r1

                center_y = self.hparams.center_y_start + (self.hparams.center_y_end - self.hparams.center_y_start) * (torch.rand(1))
                r2 = self.hparams.r2_start + ((self.hparams.r2_end - self.hparams.r2_start) * torch.rand(1)).item()
                theta = self.hparams.theta_start + ((self.hparams.theta_end - self.hparams.theta_start) * torch.rand(1)).item()
                
                grid, inverse_grid, mask = self.USR.init_grids(grid_w, grid_h, center_x, center_y, r1, r2, theta)

                grid_tensor.append(grid.unsqueeze(dim=0))
                inverse_grid_t.append(inverse_grid.unsqueeze(dim=0))
                mask_fan_t.append(mask.unsqueeze(dim=0))

            self.grid_t = torch.cat(grid_tensor).to(self.device)
            self.inverse_grid_t = torch.cat(inverse_grid_t).to(self.device)
            self.mask_fan_t = torch.cat(mask_fan_t).to(self.device)

            # Save tensors directly to the current directory
            
            torch.save(self.grid_t, grid_t_file)
            torch.save(self.inverse_grid_t, inverse_grid_t_file)
            torch.save(self.mask_fan_t, mask_fan_t_file)

            # print("Grids SAVED!")
            # print(self.grid_t.shape, self.inverse_grid_t.shape, self.mask_fan_t.shape)
        
        elif not hasattr(self, 'grid_t'):
            # Load tensors directly from the current directory
            self.grid_t = torch.load(grid_t_file).to(self.device)
            self.inverse_grid_t = torch.load(inverse_grid_t_file).to(self.device)
            self.mask_fan_t = torch.load(mask_fan_t_file).to(self.device)
    
    def compute_loss(self, Y, X_hat, step="train", sync_dist=False):
        
        loss = self.loss_fn(Y, X_hat) # The cosine similarity loss is close to 1 if the vectores are pointing in the same direction

        loss_mean = torch.mean(loss) # This should be close to 1
        # loss_std = torch.std(loss) # This should be close to 0        
        
        loss = torch.sum(torch.square(1.0 - loss)) # We minimize 1 - cos_sim so the vector are as close as possible to each other
        
        self.log(f"{step}_loss", loss, sync_dist=sync_dist)
        self.log(f"{step}_loss_mean", loss_mean, sync_dist=sync_dist)
        # self.log(f"{step}_loss_std", loss_std, sync_dist=sync_dist)

        return loss

    def volume_sampling(self, X, X_origin, X_end, use_random=False):
        with torch.no_grad():
            simulator = self.simu0
            
            grid = None
            inverse_grid = None
            mask_fan = None

            tags = self.vs.tags

            if use_random:

                simulator_idx = np.random.choice([0, 1, 2, 3, 4], replace=False)
                simulator = getattr(self, f'simu{simulator_idx}')
                # simulator = self.simu0

                if self.hparams.num_random_sweeps > 0:
                    tags = np.random.choice(self.vs.tags, self.hparams.num_random_sweeps)
                else:
                    tags = self.vs.tags

                grid_idx = torch.randint(low=0, high=self.hparams.n_grids - 1, size=(1,))
            
                grid = self.grid_t[grid_idx]
                inverse_grid = self.inverse_grid_t[grid_idx]
                mask_fan = self.mask_fan_t[grid_idx] 

            X_sweeps = []
            X_sweeps_tags = []            

            for tag in tags:                
                
                sampled_sweep_simu = self.vs.get_sweep(X, X_origin, X_end, tag, use_random=use_random, simulator=simulator, grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan)
                r_idx = torch.randint(low=0, high=sampled_sweep_simu.shape[3], size=(self.hparams.time_steps,))
                r_idx = torch.sort(r_idx)[0]                
                sampled_sweep_simu = sampled_sweep_simu[:, :, :, r_idx, :, :]                
                X_sweeps.append(sampled_sweep_simu)
                X_sweeps_tags.append(self.vs.tags_dict[tag])
                
            X_sweeps = torch.cat(X_sweeps, dim=1)
            X_sweeps_tags = torch.tensor(X_sweeps_tags, device=self.device)

            return X_sweeps, X_sweeps_tags
    
    def compute_orthogonal_frame(self, pc: torch.Tensor) -> torch.Tensor:    
        """
        Given point clouds,
        returns a tensor of shape [B, 3, 3] representing an orthogonal frame [x, y, z] for each batch.
        """

        head_idx = self.head_idx.expand(pc.shape[0], -1, -1)
        belly_idx = self.belly_idx.expand(pc.shape[0], -1, -1)
        side_idx = self.side_idx.expand(pc.shape[0], -1, -1)

        pc_head_k = knn_gather(pc, head_idx).squeeze(2)
        pc_belly_k = knn_gather(pc, belly_idx).squeeze(2)
        pc_side_k = knn_gather(pc, side_idx).squeeze(2)

        points = torch.stack([torch.mean(pc_belly_k, dim=1),
                            torch.mean(pc_head_k, dim=1),
                            torch.mean(pc_side_k, dim=1)], dim=1)

        p0 = points[:, 0]
        p1 = points[:, 1]
        p2 = points[:, 2]
        
        v1 = p1 - p0
        v2 = p2 - p0

        # Normalize x (first direction)
        x = F.normalize(v1, dim=1)

        # Compute z = normalized cross(v1, v2)
        z = F.normalize(torch.cross(v1, v2, dim=1), dim=1)

        # Compute y = cross(z, x)
        y = torch.cross(z, x, dim=1)

        # Stack the vectors as rows of the rotation matrix
        frame = torch.stack([x, y, z], dim=1)  # [B, 3, 3]

        return frame, points

    def training_step(self, train_batch, batch_idx):
        X, X_origin, X_end, X_PC = train_batch

        X, rotation_matrices = self.vs.random_rotate_3d_batch(X)

        x_sweeps, sweeps_tags = self.volume_sampling(X, X_origin, X_end, use_random=True)

        x_hat = self(x_sweeps, sweeps_tags)        

        X_PC = self.vs.apply_batch_rotation(X_PC, rotation_matrices[:,0:3,0:3])
        
        y, y_p = self.compute_orthogonal_frame(X_PC)        

        return self.compute_loss(Y=y, X_hat=x_hat, step="train")
        

    def validation_step(self, val_batch, batch_idx):
        
        X, X_origin, X_end, X_PC = val_batch

        x_sweeps, sweeps_tags = self.volume_sampling(X, X_origin, X_end)

        x_hat = self(x_sweeps, sweeps_tags)

        y, y_p = self.compute_orthogonal_frame(X_PC)        

        return self.compute_loss(Y=y, X_hat=x_hat, step="val", sync_dist=True)

        

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        """
        Forwards an image through the spatial encoder, obtaining the latent mean and sigma representations.

        Args:
            x: BxCx[SPATIAL DIMS] tensor

        """
        z = []
        for x_chunk in x.chunk(self.hparams.n_chunks_e, dim=2):
            z.append(self.encoder(x_chunk))
        z = torch.cat(z, dim=2)

        return z

    def forward(self, x_sweeps: torch.tensor, sweeps_tags: torch.tensor):
        
        batch_size = x_sweeps.shape[0]
        # x_sweeps shape is B, N, C, T, H, W. N for number of sweeps ex. torch.Size([2, 2, 1, 200, 256, 256]) 
        # tags shape torch.Size([2, 2])
        Nsweeps = x_sweeps.shape[1] # Number of sweeps -> T

        z = []
        x_v = []

        for n in range(Nsweeps):
            x_sweeps_n = x_sweeps[:, n, :, :, :, :] # [BS, C, T, H, W]
            
            tag = sweeps_tags[n]
            # x_sweeps_n = self.vs.embed_sweep(tag, x_sweeps_n)

            z_ = self.encode(x_sweeps_n) # [BS, T, self.hparams.features]

            # tag = sweeps_tags[n]
            p_enc = self.p_encoding[tag].unsqueeze(0)

            z_ = z_.permute(0, 2, 1) # Permute the time dim with the output features. -> Shape is now [BS, T, F]
            z_ = z_ + p_enc # [BS, T, self.hparams.features]            
            z_ = self.proj(z_) # [BS, self.hparams.n_chunks, self.hparams.embed_dim]

            z_ = self.attn_chunk(z_) # [BS, self.hparams.n_chunks, self.hparams.features]
            z_ = self.mha(z_)
            
            z.append(z_)

        z = torch.stack(z, dim=1) # [BS, N, self.hparams.n_chunks, 64*64*self.hparams.latent_channels]
        z = z.view(batch_size, -1, self.hparams.embed_dim).contiguous()

        z, z_s = self.attn(z, z)

        x_hat = self.proj_final(z)        
        x_hat = x_hat.view(batch_size, 3, 3).contiguous() # [BS, 3, 3]
        x_hat = F.normalize(x_hat, dim=2)

        return x_hat

    # Given diffusion model: model(x, t, cond)
# x: noisy input at timestep t
# cond: conditional information, cond=None is unconditional

def guided_sampling(model, x_t, t, cond, guidance_scale):
    # Predict noise conditioned on cond
    eps_cond = model(x_t, t, cond)
    
    # Predict noise without conditioning (unconditional)
    eps_uncond = model(x_t, t, cond=None)
    
    # Perform classifier-free guidance
    eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    
    return eps_guided


class USSeg(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.vs = VolumeSamplingBlindSweep()

        self.model = monai_nets.UNet(
            spatial_dims=self.hparams.spatial_dims,
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=self.hparams.norm,
        )
        self.loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
        # self.post_pred = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True, to_onehot=2)])
        # self.post_label = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=2)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

        self.simu0 = TimeDistributed(us_simulation_jit.MergedLinearCutLabel11(), time_dim=2)
        self.simu1 = TimeDistributed(us_simulation_jit.MergedCutLabel11(), time_dim=2)
        self.simu2 = TimeDistributed(us_simulation_jit.MergedUSRLabel11(), time_dim=2)
        self.simu3 = TimeDistributed(us_simulation_jit.MergedLinearLabel11(), time_dim=2)
        self.simu4 = TimeDistributed(us_simulation_jit.MergedLinearLabel11WOG(), time_dim=2)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("Calculate baby frame of orientation")

        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
        # Image Encoder parameters 
        group.add_argument("--spatial_dims", type=int, default=3, help='Spatial dimensions for the encoder')
        group.add_argument("--in_channels", type=int, default=1, help='Input number of channels')
        group.add_argument("--out_channels", type=int, default=12, help='Output number of channels')
        group.add_argument("--time_steps", type=int, default=128, help='Sample N number of frames from sweep')
        group.add_argument("--norm", type=str, default='BATCH', help='Type of norm')
        group.add_argument("--num_sweeps", type=int, default=1, help='Sample N sweeps from the volume')
        group.add_argument("--n_grids", type=int, default=200, help='Number of grids')
        
        
        # group.add_argument("--p_ids", type=int, nargs='+', default=[1470, 3369, 2043], help='Point ids to compute the orthogonal orientation frame')
        # group.add_argument("--target_label", type=int, default=7, help='Target label')

        # group.add_argument("--n_fixed_samples", type=int, default=12288, help='Number of fixed samples')

        return parent_parser
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer

    def on_fit_start(self):

        # Define the file names directly without using out_dir
        grid_t_file = 'grid_t.pt'
        inverse_grid_t_file = 'inverse_grid_t.pt'
        mask_fan_t_file = 'mask_fan_t.pt'

        if not os.path.exists(grid_t_file):
            grid_tensor = []
            inverse_grid_t = []
            mask_fan_t = []

            for i in range(self.hparams.n_grids):

                grid_w, grid_h = self.hparams.grid_w, self.hparams.grid_h
                center_x = self.hparams.center_x
                r1 = self.hparams.r1

                center_y = self.hparams.center_y_start + (self.hparams.center_y_end - self.hparams.center_y_start) * (torch.rand(1))
                r2 = self.hparams.r2_start + ((self.hparams.r2_end - self.hparams.r2_start) * torch.rand(1)).item()
                theta = self.hparams.theta_start + ((self.hparams.theta_end - self.hparams.theta_start) * torch.rand(1)).item()
                
                grid, inverse_grid, mask = self.USR.init_grids(grid_w, grid_h, center_x, center_y, r1, r2, theta)

                grid_tensor.append(grid.unsqueeze(dim=0))
                inverse_grid_t.append(inverse_grid.unsqueeze(dim=0))
                mask_fan_t.append(mask.unsqueeze(dim=0))

            self.grid_t = torch.cat(grid_tensor).to(self.device)
            self.inverse_grid_t = torch.cat(inverse_grid_t).to(self.device)
            self.mask_fan_t = torch.cat(mask_fan_t).to(self.device)

            # Save tensors directly to the current directory
            
            torch.save(self.grid_t, grid_t_file)
            torch.save(self.inverse_grid_t, inverse_grid_t_file)
            torch.save(self.mask_fan_t, mask_fan_t_file)

            # print("Grids SAVED!")
            # print(self.grid_t.shape, self.inverse_grid_t.shape, self.mask_fan_t.shape)
        
        elif not hasattr(self, 'grid_t'):
            # Load tensors directly from the current directory
            self.grid_t = torch.load(grid_t_file).to(self.device)
            self.inverse_grid_t = torch.load(inverse_grid_t_file).to(self.device)
            self.mask_fan_t = torch.load(mask_fan_t_file).to(self.device)
    
    def compute_loss(self, Y, X_hat, step="train", sync_dist=False):
        
        loss = self.loss_fn(X_hat, Y) 
        
        self.log(f"{step}_loss", loss, sync_dist=sync_dist)
        # self.log(f"{step}_loss_mean", loss_mean, sync_dist=sync_dist)
        # self.log(f"{step}_loss_std", loss_std, sync_dist=sync_dist)

        return loss

    def volume_sampling(self, X, X_origin, X_end, use_random=False):
        with torch.no_grad():
            simulator = self.simu0
            
            grid = None
            inverse_grid = None
            mask_fan = None

            tags = self.vs.tags

            if use_random:

                simulator_idx = np.random.choice([0, 1, 2, 3, 4], replace=False)
                simulator = getattr(self, f'simu{simulator_idx}')

                grid_idx = torch.randint(low=0, high=self.hparams.n_grids - 1, size=(1,))
            
                grid = self.grid_t[grid_idx]
                inverse_grid = self.inverse_grid_t[grid_idx]
                mask_fan = self.mask_fan_t[grid_idx] 

            
            tags = np.random.choice(self.vs.tags, self.hparams.num_sweeps)
                

            X_sweeps = []
            Y_sweeps = []
            X_sweeps_tags = []            

            for tag in tags:                
                
                sampled_sweep_simu, sampled_sweep = self.vs.get_sweep(X, X_origin, X_end, tag, use_random=use_random, simulator=simulator, grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan, return_masked=True)
                
                r_idx = torch.randint(low=0, high=sampled_sweep_simu.shape[3], size=(self.hparams.time_steps,))
                r_idx = torch.sort(r_idx)[0]
                sampled_sweep_simu = sampled_sweep_simu[:, :, :, r_idx, :, :]
                sampled_sweep = sampled_sweep[:, :, r_idx, :, :]        
                X_sweeps.append(sampled_sweep_simu)
                X_sweeps_tags.append(self.vs.tags_dict[tag])

                Y_sweeps.append(sampled_sweep)
                
            X_sweeps = torch.cat(X_sweeps, dim=1)
            X_sweeps_tags = torch.tensor(X_sweeps_tags, device=self.device)
            Y_sweeps = torch.cat(Y_sweeps, dim=1)

            return X_sweeps, Y_sweeps, X_sweeps_tags

    def training_step(self, train_batch, batch_idx):
        X, X_origin, X_end, X_PC = train_batch

        X, rotation_matrices = self.vs.random_rotate_3d_batch(X)

        x, y, sweeps_tags = self.volume_sampling(X, X_origin, X_end, use_random=True)
        x = x.squeeze(1)

        x_hat = self(x)

        return self.compute_loss(Y=y, X_hat=x_hat, step="train")
        

    def validation_step(self, val_batch, batch_idx):
        
        X, X_origin, X_end, X_PC = val_batch

        x, y, sweeps_tags = self.volume_sampling(X, X_origin, X_end)
        x = x.squeeze(1)

        x_hat = self(x)

        self.dice_metric(y=y, y_pred=x_hat)

        return self.compute_loss(Y=y, X_hat=x_hat, step="val", sync_dist=True)

    def on_validation_epoch_end(self):
        # aggregate the final mean dice result
        metric = self.dice_metric.aggregate().item()

        self.log("val_dice", metric, sync_dist=True)
        
        self.dice_metric.reset()

    def forward(self, x_sweeps: torch.tensor):
        return self.model(x_sweeps)