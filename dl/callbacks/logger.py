from lightning.pytorch.callbacks import Callback
import torchvision
import torch

import matplotlib.pyplot as plt

import numpy as np 

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from torch.nn import functional as F

class MocoImageLogger(Callback):
    def __init__(self, num_images=12, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            (img1, img2), _ = batch
            
            max_num_image = min(img1.shape[0], self.num_images)
            grid_img1 = torchvision.utils.make_grid(img1[0:max_num_image])
            trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
            
            grid_img2 = torchvision.utils.make_grid(img2[0:max_num_image])
            trainer.logger.experiment.add_image('img2', grid_img2.cpu().numpy(), 0)
            with torch.no_grad():
                logits, labels, k = pl_module(img1, img2, pl_module.queue)

            for idx, logit in enumerate(logits[0:max_num_image]):
                trainer.logger.experiment.add_scalar('logits', torch.argmax(logit).cpu().numpy(), idx)


class SimCLRImageLogger(Callback):
    def __init__(self, num_images=12, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            (img1, img2), _ = batch
            
            max_num_image = min(img1.shape[0], self.num_images)
            grid_img1 = torchvision.utils.make_grid(img1[0:max_num_image])
            trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
            
            grid_img2 = torchvision.utils.make_grid(img2[0:max_num_image])
            trainer.logger.experiment.add_image('img2', grid_img2.cpu().numpy(), 0)

class SimImageLogger(Callback):
    def __init__(self, num_images=18, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            img1, img2 = batch
            
            with torch.no_grad():
                img2 = pl_module.noise_transform(img2)

            max_num_image = min(img1.shape[0], self.num_images)
            grid_img1 = torchvision.utils.make_grid(img1[0:max_num_image])
            trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
            
            grid_img2 = torchvision.utils.make_grid(img2[0:max_num_image])
            trainer.logger.experiment.add_image('img2', grid_img2.cpu().numpy(), 0)

class SimScoreImageLogger(Callback):
    def __init__(self, num_images=18, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            (img1, img2), scores = batch
            
            with torch.no_grad():
                img2 = pl_module.noise_transform(img2)

            max_num_image = min(img1.shape[0], self.num_images)
            grid_img1 = torchvision.utils.make_grid(img1[0:max_num_image])


            # Generate figure
            plt.clf()
            fig = plt.figure(figsize=(7, 9))
            ax = plt.imshow(grid_img1.permute(1, 2, 0).cpu().numpy())            
            # trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
            trainer.logger.experiment["images"].upload(fig)

            plt.close()
            
            # grid_img2 = torchvision.utils.make_grid(img2[0:max_num_image])
            # trainer.logger.experiment.add_image('img2', grid_img2.cpu().numpy(), 0)

            # for idx, s in enumerate(scores):
            #     trainer.logger.experiment.add_scalar('scores', s, idx)

class SimNorthImageLogger(Callback):
    def __init__(self, num_images=18, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            img1, img2 = batch
            
            with torch.no_grad():
                img2 = pl_module.noise_transform(img2)

            max_num_image = min(img1.shape[0], self.num_images)
            grid_img1 = torchvision.utils.make_grid(img1[0:max_num_image])


            # Generate figure
            plt.clf()
            fig = plt.figure(figsize=(7, 9))
            ax = plt.imshow(grid_img1.permute(1, 2, 0).cpu().numpy())

            # trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
            trainer.logger.experiment["images"].upload(fig)

            plt.close()
            
            # grid_img2 = torchvision.utils.make_grid(img2[0:max_num_image])
            # trainer.logger.experiment.add_image('img2', grid_img2.cpu().numpy(), 0)

            # for idx, s in enumerate(scores):
            #     trainer.logger.experiment.add_scalar('scores', s, idx)

class EffnetDecodeImageLogger(Callback):
    def __init__(self, num_images=12, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            img1, img2 = pl_module.train_transform(batch)
            
            max_num_image = min(img1.shape[0], self.num_images)
            grid_img1 = torchvision.utils.make_grid(img1[0:max_num_image])
            trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
            
            grid_img2 = torchvision.utils.make_grid(img2[0:max_num_image])
            trainer.logger.experiment.add_image('img2', grid_img2.cpu().numpy(), 0)

            # with torch.no_grad():
            #     x_hat, z = pl_module(img1)

            #     grid_x_hat = torchvision.utils.make_grid(x_hat[0:max_num_image])
            #     trainer.logger.experiment.add_image('x_hat', grid_x_hat, 0)

class AutoEncoderImageLogger(Callback):
    def __init__(self, num_images=16, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():
                img1, img2 = batch
                img2 = pl_module.noise_transform(img2)

                max_num_image = min(img1.shape[0], self.num_images)
                grid_img1 = torchvision.utils.make_grid(img1[0:max_num_image])
                trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
                
                grid_img2 = torchvision.utils.make_grid(img2[0:max_num_image])
                trainer.logger.experiment.add_image('img2', grid_img2.cpu().numpy(), 0)

                x_hat, z = pl_module(img2)

                grid_x_hat = torchvision.utils.make_grid(x_hat[0:max_num_image])
                trainer.logger.experiment.add_image('x_hat', torch.tensor(grid_x_hat), 0)

class BlindSweepImageLogger(Callback):
    def __init__(self, num_images=16, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            
            img, ga = batch                
            
            # grid_img1 = torchvision.utils.make_grid(img[0,:,:,:])
            # trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)


class DiffusionImageLogger(Callback):
    def __init__(self, num_images=16, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():
                x = batch

                max_num_image = min(x.shape[0], self.num_images)
                grid_img1 = torchvision.utils.make_grid(x[0:max_num_image])


                trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), pl_module.global_step)

                x_hat, z_mu, z_sigma = pl_module(x)

                grid_x_hat = torchvision.utils.make_grid(x_hat[0:max_num_image])


                trainer.logger.experiment.add_image('x_hat', torch.tensor(grid_x_hat), pl_module.global_step)
                

class DiffusionImageLoggerNeptune(Callback):
    def __init__(self, num_images=16, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():
                x = batch

                max_num_image = min(x.shape[0], self.num_images)


                x = x[0:max_num_image]

                grid_img1 = torchvision.utils.make_grid(x[0:max_num_image])
                x_ = pl_module(x)

                if isinstance(x_, tuple):
                    if len(x_) == 2:
                        x_hat, _ = x_
                    else:
                        x_hat, z_mu, z_sigma = x_
                else:
                    x_hat = x_


                x = x.clip(0, 1)
                x_hat = x_hat.clip(0,1)

                grid_x_hat = torchvision.utils.make_grid(x_hat[0:max_num_image])
                
                # Generate figure                
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_img1.permute(1, 2, 0).cpu().numpy())
                # trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
                trainer.logger.experiment["images/x"].upload(fig)
                plt.close()

                # Generate figure
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_hat.permute(1, 2, 0).cpu().numpy())
                # trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
                trainer.logger.experiment["images/x_hat"].upload(fig)
                plt.close()


                if hasattr(pl_module, 'autoencoderkl'):
                    z_mu, z_sigma = pl_module.autoencoderkl.encode(x)
                    z_mu = z_mu.detach()
                    z_sigma = z_sigma.detach()
                    z_vae = pl_module.autoencoderkl.sampling(z_mu, z_sigma)

                    z_vae = z_vae.clip(0,1)
                    z_mu = z_mu.clip(0, 1)

                    grid_x_z_mu = torchvision.utils.make_grid(z_mu[0:max_num_image])

                    fig = plt.figure(figsize=(7, 9))
                    ax = plt.imshow(grid_x_z_mu.permute(1, 2, 0).cpu().numpy())
                    trainer.logger.experiment["images/z_mu"].upload(fig)
                    plt.close()

                    grid_x_z_vae = torchvision.utils.make_grid(z_vae[0:max_num_image])

                    fig = plt.figure(figsize=(7, 9))
                    ax = plt.imshow(grid_x_z_vae.permute(1, 2, 0).cpu().numpy())
                    trainer.logger.experiment["images/z_vae"].upload(fig)
                    plt.close()
                


class DiffusionImageLoggerPairedNeptune(Callback):
    def __init__(self, num_images=16, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():
                x = batch[0]
                y = batch[1]

                max_num_image = min(x.shape[0], self.num_images)


                x = x[0:max_num_image]

                grid_img1 = torchvision.utils.make_grid(x[0:max_num_image])
                x_ = pl_module(x)

                if isinstance(x_, tuple):
                    if len(x_) == 2:
                        x_hat, _ = x_
                    else:
                        x_hat, z_mu, z_sigma = x_
                else:
                    x_hat = x_


                x = x.clip(0, 1)
                x_hat = x_hat.clip(0,1)

                grid_x_hat = torchvision.utils.make_grid(x_hat[0:max_num_image])
                
                # Generate figure                
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_img1.permute(1, 2, 0).cpu().numpy())
                # trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
                trainer.logger.experiment["images/x"].upload(fig)
                plt.close()

                # Generate figure
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_hat.permute(1, 2, 0).cpu().numpy())
                # trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
                trainer.logger.experiment["images/x_hat"].upload(fig)
                plt.close()



                y = y[0:max_num_image]

                grid_img2 = torchvision.utils.make_grid(y[0:max_num_image])
                
                # Generate figure
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_img2.permute(1, 2, 0).cpu().numpy())                
                trainer.logger.experiment["images/y"].upload(fig)
                plt.close()

class GenerativeImageLoggerNeptune(Callback):
    def __init__(self, num_images=16, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():
                x = batch

                max_num_image = min(x.shape[0], self.num_images)

                x = x[0:max_num_image]

                x_hat = pl_module(x.shape[0])

                x = x - torch.min(x)
                x = x/torch.max(x)

                x_hat = x_hat - torch.min(x_hat)
                x_hat = x_hat/torch.max(x_hat)


                grid_img1 = torchvision.utils.make_grid(x)
                grid_x_hat = torchvision.utils.make_grid(x_hat)
                
                # Generate figure                
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_img1.permute(1, 2, 0).cpu().numpy())
                # trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
                trainer.logger.experiment["images/x"].upload(fig)
                plt.close()

                # Generate figure
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_hat.permute(1, 2, 0).cpu().numpy())
                # trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
                trainer.logger.experiment["images/x_hat"].upload(fig)
                plt.close()

class DiffusionImageLoggerMRUS(Callback):
    def __init__(self, num_images=16, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():
                x_mr, x_us = batch

                max_num_image = min(x_mr.shape[0], self.num_images)

                x_mr_us_hat, z_mu_mr_us, z_sigma_mr_us = pl_module.get_us(x_mr)

                x_mr = x_mr.clip(0, 1)
                x_mr_us_hat = x_mr_us_hat.clip(0,1)


                grid_x_mr = torchvision.utils.make_grid(x_mr[0:max_num_image])
                grid_x_mr_us_hat = torchvision.utils.make_grid(x_mr_us_hat[0:max_num_image])

                
                # add figure          
                trainer.logger.experiment.add_image('x_mr', grid_x_mr.cpu().numpy(), pl_module.global_step)
                trainer.logger.experiment.add_image('x_mr_us_hat', grid_x_mr_us_hat.cpu().numpy(), pl_module.global_step)


                x_us_mr_hat, z_mu_us_mr, z_sigma_us_mr = pl_module.get_mr(x_us)

                x_us = x_us.clip(0, 1)
                x_us_mr_hat = x_us_mr_hat.clip(0,1)


                grid_x_us = torchvision.utils.make_grid(x_us_mr_hat[0:max_num_image])
                grid_x_us_mr_hat = torchvision.utils.make_grid(x_us_mr_hat[0:max_num_image])

                
                # Generate figure                
                trainer.logger.experiment.add_image('x_us', grid_x_us.cpu().numpy(), pl_module.global_step)
                trainer.logger.experiment.add_image('x_us_mr_hat', grid_x_us_mr_hat.cpu().numpy(), pl_module.global_step)
                


class DiffusionImageLoggerMRUSNeptune(Callback):
    def __init__(self, num_images=16, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():
                x_mr, x_us = batch

                max_num_image = min(x_mr.shape[0], self.num_images)

                x_mr_us_hat, z_mu_mr_us, z_sigma_mr_us = pl_module.get_us(x_mr)

                x_mr = x_mr.clip(0, 1)
                x_mr_us_hat = x_mr_us_hat.clip(0,1)


                grid_x_mr = torchvision.utils.make_grid(x_mr[0:max_num_image])
                grid_x_mr_us_hat = torchvision.utils.make_grid(x_mr_us_hat[0:max_num_image])

                
                # Generate figure                
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_mr.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_mr"].upload(fig)
                plt.close()

                # Generate figure
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_mr_us_hat.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_mr_us_hat"].upload(fig)
                plt.close()





                x_us_mr_hat, z_mu_us_mr, z_sigma_us_mr = pl_module.get_mr(x_us)

                x_us = x_us.clip(0, 1)
                x_us_mr_hat = x_us_mr_hat.clip(0,1)


                grid_x_us = torchvision.utils.make_grid(x_us[0:max_num_image])
                grid_x_us_mr_hat = torchvision.utils.make_grid(x_us_mr_hat[0:max_num_image])

                
                # Generate figure                
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_us.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_us"].upload(fig)
                plt.close()

                # Generate figure
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_us_mr_hat.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_us_mr_hat"].upload(fig)
                plt.close()


class ImageLoggerMustUSNeptune(Callback):
    def __init__(self, num_images=16, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():
                x_a, x_b = batch

                max_num_image = min(x_a.shape[0], self.num_images)

                fake_b = pl_module.get_b(x_a)

                fake_b = torch.clip(fake_b, min=0.0, max=1.0)
                fake_b = torch.clip(fake_b, min=0.0, max=1.0)


                grid_x_a = torchvision.utils.make_grid(x_a[0:max_num_image])
                grid_fake_b = torchvision.utils.make_grid(fake_b[0:max_num_image])
                
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_a.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_a"].upload(fig)
                plt.close()

                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_fake_b.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/fake_b"].upload(fig)
                plt.close()


                fake_a = pl_module.get_a(x_b)

                fake_a = torch.clip(fake_a, min=0.0, max=1.0)

                grid_x_b = torchvision.utils.make_grid(x_b[0:max_num_image])
                grid_fake_a = torchvision.utils.make_grid(fake_a[0:max_num_image])
                
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_b.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_b"].upload(fig)
                plt.close()
                
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_fake_a.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/fake_a"].upload(fig)
                plt.close()


class ImageLoggerLotusNeptune(Callback):
    def __init__(self, num_images=12, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():
                X, X_origin, X_end = batch
        
                X_label, Y_simu, grid, inverse_grid, mask_fan  = pl_module.get_sweeps(X, X_origin, X_end)

                repeats = [1,]*len(X_label.shape)
                repeats[0] = X_label.shape[0]
                grid = grid.repeat(repeats)
                inverse_grid = inverse_grid.repeat(repeats)
                mask_fan = mask_fan.repeat(repeats)
                
                X_simu = pl_module(X_label, grid, inverse_grid, mask_fan)

                rt_idx = torch.randint(low=0, high=X_simu.shape[0], size=(self.num_images,))
                fig = px.imshow(X_simu[rt_idx].squeeze().cpu().numpy(), animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
                # ## Log interactive chart to Neptune
                trainer.logger.experiment["images/simu"].upload(fig)

                fig = px.imshow(Y_simu[rt_idx].squeeze().cpu().numpy(), animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
                # ## Log interactive chart to Neptune
                trainer.logger.experiment["images/target"].upload(fig)


class ImageLoggerLotusV2Neptune(Callback):
    def __init__(self, num_images=6, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():
                x_label, x_us = batch

                x_diffusor = x_label['img']
                x_label = x_label['seg']

                max_num_image = min(x_label.shape[0], self.num_images)

                x_lotus_us = pl_module(x_label)
                x_lotus_us = torch.clip(x_lotus_us, min=0.0, max=1.0)


                x_label = x_label/torch.max(x_label)
                grid_x_label = torchvision.utils.make_grid(x_label[0:max_num_image])
                grid_x_lotus_us = torchvision.utils.make_grid(x_lotus_us[0:max_num_image])
                
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_label.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_label"].upload(fig)
                plt.close()

                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_lotus_us.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_lotus_us"].upload(fig)
                plt.close()

                
                grid_x_diffusor = torchvision.utils.make_grid(x_diffusor[0:max_num_image])                
                
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_diffusor.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_diffusor"].upload(fig)
                plt.close()
                
                
                grid_x_us = torchvision.utils.make_grid(x_us[0:max_num_image])                
                
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_us.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_us"].upload(fig)
                plt.close()
                    

class UltrasoundRenderingDiffLogger(Callback):
    def __init__(self, num_images=8, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():
                x_label, x_us = batch
                
                x_diff = x_label['img']
                x_seg = x_label['seg']

                max_num_image = min(x_seg.shape[0], self.num_images)

                x_lotus_us = pl_module.us_renderer(x_seg)[0]
                x_lotus_us = torch.clip(x_lotus_us, min=0.0, max=1.0)
                x_lotus_recon = pl_module(x_seg)
                x_lotus_recon = torch.clip(x_lotus_recon, min=0.0, max=1.0)

                x_seg = x_seg/torch.max(x_seg)
                grid_x_label = torchvision.utils.make_grid(x_seg[0:max_num_image])
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_label.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_label"].upload(fig)
                plt.close()


                grid_x_lotus_us = torchvision.utils.make_grid(x_lotus_us[0:max_num_image])
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_lotus_us.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_lotus_us"].upload(fig)
                plt.close()

                grid_x_lotus_recon = torchvision.utils.make_grid(x_lotus_recon[0:max_num_image])
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_lotus_recon.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_lotus_recon_us"].upload(fig)
                plt.close()

                grid_x_us = torchvision.utils.make_grid(x_us[0:max_num_image])                
                
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_us.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_us"].upload(fig)
                plt.close()

class CutLogger(Callback):
    def __init__(self, num_images=8, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():

                labeled, x_us = batch
                max_num_image = min(x_us.shape[0], self.num_images)
                
                if isinstance(labeled, dict):
                    x_diff = labeled['img']
                    x_seg = labeled['seg']

                    x_diff = torch.clip(x_diff, min=0.0, max=1.0)/torch.max(x_diff)                

                    grid_x_diff = torchvision.utils.make_grid(x_diff[0:max_num_image], nrow=4)
                    fig = plt.figure(figsize=(7, 9))
                    ax = plt.imshow(grid_x_diff.permute(1, 2, 0).cpu().numpy())
                    trainer.logger.experiment["images/x_diffusor"].upload(fig)
                    plt.close()

                else:                    
                    x_seg = labeled  

                grid_idx = torch.randint(low=0, high=pl_module.hparams.n_grids - 1, size=(x_us.shape[0],))
        
                grid = pl_module.grid_t[grid_idx]
                inverse_grid = pl_module.inverse_grid_t[grid_idx]
                mask_fan = pl_module.mask_fan_t[grid_idx]
                  

                x_lotus_us = pl_module.USR(x_seg, grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan)

                if hasattr(pl_module, 'autoencoderkl'):
                    x_lotus_fake = pl_module(x_seg)
                else:
                    x_lotus_fake = pl_module.G(pl_module.transform_us(x_lotus_us))

                    if isinstance(x_lotus_fake, tuple):
                        x_lotus_fake = x_lotus_fake[0]

                    x_lotus_fake = x_lotus_fake*pl_module.transform_us(mask_fan)
                
                x_lotus_us = x_lotus_us - torch.min(x_lotus_us)
                x_lotus_us = x_lotus_us/torch.max(x_lotus_us)
                x_lotus_fake = torch.clip(x_lotus_fake, min=0.0, max=1.0)

                x_seg = x_seg/torch.max(x_seg)
                grid_x_label = torchvision.utils.make_grid(x_seg[0:max_num_image], nrow=4)
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_label.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_label"].upload(fig)
                plt.close()


                grid_x_lotus_us = torchvision.utils.make_grid(x_lotus_us[0:max_num_image], nrow=4)
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_lotus_us.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_lotus_us"].upload(fig)
                plt.close()

                grid_x_lotus_fake = torchvision.utils.make_grid(x_lotus_fake[0:max_num_image], nrow=4)
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_lotus_fake.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_lotus_fake"].upload(fig)
                plt.close()

                grid_x_us = torchvision.utils.make_grid(x_us[0:max_num_image], nrow=4)                
                
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_us.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_us"].upload(fig)
                plt.close()

class CutGLogger(Callback):
    def __init__(self, num_images=4, log_steps=100, *args, **kwargs):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():

                X, Y = batch

                max_num_image = min(X.shape[0], self.num_images)

                Y_fake = pl_module(X)
                
                Y_fake = torch.clip(Y_fake, min=0.0, max=1.0)

                grid_x = torchvision.utils.make_grid(X[0:max_num_image], nrow=2)
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x"].upload(fig)
                plt.close()

                grid_y_fake = torchvision.utils.make_grid(Y_fake[0:max_num_image], nrow=2)
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_y_fake.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/y_fake"].upload(fig)
                plt.close()

                grid_y = torchvision.utils.make_grid(Y[0:max_num_image], nrow=2)
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_y.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/y"].upload(fig)
                plt.close()


class CutGConditionalLogger(Callback):
    def __init__(self, num_images=4, log_steps=100, *args, **kwargs):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():

                X, X_labels, Y, Y_labels = batch    

                max_num_image = min(X.shape[0], self.num_images)

                Y_fake = pl_module(X, Y_labels)
                
                Y_fake = torch.clip(Y_fake, min=0.0, max=1.0)

                grid_x = torchvision.utils.make_grid(X[0:max_num_image], nrow=2)
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x"].upload(fig)
                plt.close()

                grid_y_fake = torchvision.utils.make_grid(Y_fake[0:max_num_image], nrow=2)
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_y_fake.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/y_fake"].upload(fig)
                plt.close()

                grid_y = torchvision.utils.make_grid(Y[0:max_num_image], nrow=2)
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_y.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/y"].upload(fig)
                plt.close()
                


class SPADELogger(Callback):
    def __init__(self, num_images=8, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():

                labeled, x_us = batch
                x_diff = labeled['img']
                x_seg = labeled['seg']        

                max_num_image = min(x_seg.shape[0], self.num_images)


                grid_idx = torch.randint(low=0, high=pl_module.hparams.n_grids - 1, size=(x_seg.shape[0],))
        
                grid = pl_module.grid_t[grid_idx]
                inverse_grid = pl_module.inverse_grid_t[grid_idx]
                mask_fan = pl_module.mask_fan_t[grid_idx]

                x_lotus_us = pl_module.USR(x_seg, grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan)
                x_seg = pl_module.transform_us(x_seg*mask_fan)
                
                labels = pl_module.one_hot(x_seg).to(pl_module.device)

                x_lotus_fake, _, _ = pl_module.G(pl_module.transform_us(x_lotus_us), labels)
                x_lotus_fake = x_lotus_fake*pl_module.transform_us(mask_fan)

                x_lotus_us = x_lotus_us - torch.min(x_lotus_us)
                x_lotus_us = x_lotus_us/torch.max(x_lotus_us)
                x_lotus_fake = torch.clip(x_lotus_fake, min=0.0, max=1.0)
                
                x_diff = torch.clip(x_diff, min=0.0, max=1.0)/torch.max(x_diff)                

                grid_x_diff = torchvision.utils.make_grid(x_diff[0:max_num_image], nrow=4)
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_diff.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_diffusor"].upload(fig)
                plt.close()

                x_seg = x_seg/torch.max(x_seg)
                grid_x_label = torchvision.utils.make_grid(x_seg[0:max_num_image], nrow=4)
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_label.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_label"].upload(fig)
                plt.close()


                grid_x_lotus_us = torchvision.utils.make_grid(x_lotus_us[0:max_num_image], nrow=4)
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_lotus_us.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_lotus_us"].upload(fig)
                plt.close()

                grid_x_lotus_fake = torchvision.utils.make_grid(x_lotus_fake[0:max_num_image], nrow=4)
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_lotus_fake.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_lotus_fake"].upload(fig)
                plt.close()

                grid_x_us = torchvision.utils.make_grid(x_us[0:max_num_image], nrow=4)                
                
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_us.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_us"].upload(fig)
                plt.close()


class USAEReconstructionNeptuneLogger(Callback): 
    def __init__(self, num_images=1, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():

                X, X_origin, X_end = batch
                
                grid_idx = torch.randint(low=0, high=pl_module.grid_t.shape[0] - 1, size=(X.shape[0],))
                grid = pl_module.grid_t[grid_idx]
                inverse_grid = pl_module.inverse_grid_t[grid_idx]
                mask_fan = pl_module.mask_fan_t[grid_idx]

                X_sweeps = []
                for tag in np.random.choice(pl_module.vs.tags, 2):
                    sampled_sweep = pl_module.vs.diffusor_sampling_tag(tag, X.to(torch.float), X_origin.to(torch.float), X_end.to(torch.float))
                    sampled_sweep_simu = torch.cat([pl_module.us_simulator_cut_td(ss.unsqueeze(dim=0), grid, inverse_grid, mask_fan) for ss in sampled_sweep], dim=0)
                    sampled_sweep_in_fov = pl_module.vs.simulated_sweep_in_fov(tag, sampled_sweep_simu.detach())

                    X_sweeps.append(sampled_sweep_in_fov.unsqueeze(1).unsqueeze(1)) #Add time dimension and channel
                X_sweeps = torch.cat(X_sweeps, dim=1)
                
                Y = pl_module.us_simulator_cut_td.module.USR.mean_diffusor_dict[X.to(torch.long)].to(pl_module.device) + torch.randn(X.shape, device=pl_module.device) * pl_module.us_simulator_cut_td.module.USR.variance_diffusor_dict[X.to(torch.long)].to(pl_module.device)
                Y = pl_module.vs.diffusor_in_fov(Y, X_origin, X_end)

                reconstruction, _, _ = pl_module(X_sweeps)

                fig = px.imshow(X_sweeps[0][0].squeeze().cpu().numpy(), animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
                # ## Log interactive chart to Neptune
                trainer.logger.experiment["images/sweep"].upload(fig)

                fig = px.imshow(Y[0].squeeze().cpu().numpy(), animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
                # ## Log interactive chart to Neptune
                trainer.logger.experiment["images/target"].upload(fig)
                
                fig = px.imshow(reconstruction[0].squeeze().cpu().numpy(), animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
                # ## Log interactive chart to Neptune
                trainer.logger.experiment["images/reconstruction"].upload(fig)

class USVQVAEReconstructionNeptuneLogger(Callback): 
    def __init__(self, num_images=1, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():

                X, X_origin, X_end = batch
                
                grid_idx = torch.randint(low=0, high=pl_module.grid_t.shape[0] - 1, size=(X.shape[0],))
                grid = pl_module.grid_t[grid_idx]
                inverse_grid = pl_module.inverse_grid_t[grid_idx]
                mask_fan = pl_module.mask_fan_t[grid_idx]

                X_sweeps, Y = pl_module.volume_sampling(X, X_origin, X_end, grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan)

                reconstruction, _ = pl_module(X_sweeps)

                rt_idx = torch.randint(low=0, high=X_sweeps.shape[1], size=(1,))
                fig = px.imshow(X_sweeps[0][rt_idx].squeeze().cpu().numpy(), animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
                # ## Log interactive chart to Neptune
                trainer.logger.experiment["images/sweep"].upload(fig)

                fig = px.imshow(Y[0].squeeze().cpu().numpy(), animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
                # ## Log interactive chart to Neptune
                trainer.logger.experiment["images/target"].upload(fig)
                
                fig = px.imshow(reconstruction[0].squeeze().cpu().numpy(), animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
                # ## Log interactive chart to Neptune
                trainer.logger.experiment["images/reconstruction"].upload(fig)

class USUNetReconstructionNeptuneLogger(Callback): 
    def __init__(self, num_images=1, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():

                X, X_origin, X_end = batch
                
                grid_idx = torch.randint(low=0, high=pl_module.grid_t.shape[0] - 1, size=(X.shape[0],))
                grid = pl_module.grid_t[grid_idx]
                inverse_grid = pl_module.inverse_grid_t[grid_idx]
                mask_fan = pl_module.mask_fan_t[grid_idx]

                X_sweeps, Y = pl_module.volume_sampling(X, X_origin, X_end, grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan)

                reconstruction = pl_module(X_sweeps)

                rt_idx = torch.randint(low=0, high=X_sweeps.shape[1], size=(1,))
                fig = px.imshow(X_sweeps[0][rt_idx].squeeze().cpu().numpy(), animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
                # ## Log interactive chart to Neptune
                trainer.logger.experiment["images/sweep"].upload(fig)

                fig = px.imshow(Y[0].squeeze().cpu().numpy(), animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
                # ## Log interactive chart to Neptune
                trainer.logger.experiment["images/target"].upload(fig)
                
                fig = px.imshow(reconstruction[0].squeeze().cpu().numpy(), animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
                # ## Log interactive chart to Neptune
                trainer.logger.experiment["images/reconstruction"].upload(fig)

class USVQGanReconstructionLogger(Callback): 
    def __init__(self, num_images=1, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():

                X, X_origin, X_end = batch
                Y = pl_module.volume_sampling(X, X_origin, X_end)

                reconstruction, _ = pl_module(Y)

                fig = px.imshow(Y[0].squeeze().cpu().numpy(), animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
                # ## Log interactive chart to Neptune
                trainer.logger.experiment["images/target"].upload(fig)
                
                fig = px.imshow(reconstruction[0].squeeze().cpu().numpy(), animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
                # ## Log interactive chart to Neptune
                trainer.logger.experiment["images/reconstruction"].upload(fig)

class USVQGanReconstructionLabel_10Logger(Callback): 
    def __init__(self, num_images=1, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():

                X, X_origin, X_end = batch
                Y = pl_module.volume_sampling(X, X_origin, X_end)

                reconstruction, _ = pl_module(Y)

                fig = px.imshow(Y[0].squeeze().cpu().numpy(), animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
                # ## Log interactive chart to Neptune
                trainer.logger.experiment["images/target"].upload(fig)
                

                reconstruction = torch.argmax(reconstruction, dim=1).float()/reconstruction.shape[1]
                fig = px.imshow(reconstruction[0].squeeze().cpu().numpy(), animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
                # ## Log interactive chart to Neptune
                trainer.logger.experiment["images/reconstruction"].upload(fig)

class USAEReconstructionLabel_10Logger(Callback): 
    def __init__(self, num_images=1, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():

                X, X_origin, X_end = batch
                X, Y = pl_module.volume_sampling(X, X_origin, X_end)

                reconstruction, _, _ = pl_module(Y)

                fig = px.imshow(X[0].squeeze().cpu().numpy(), animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
                # ## Log interactive chart to Neptune
                trainer.logger.experiment["images/target"].upload(fig)
                

                reconstruction = torch.argmax(reconstruction, dim=1).float()
                fig = px.imshow(reconstruction[0].squeeze().cpu().numpy(), animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
                # ## Log interactive chart to Neptune
                trainer.logger.experiment["images/reconstruction"].upload(fig)

class USPCLoggerNeptune(Callback):
    # This callback logs images for visualization during training, with the ability to log images to the Neptune logging system for easy monitoring and analysis
    def __init__(self, num_surf=1, log_steps=10):
        self.log_steps = log_steps
        self.num_surf = num_surf
        self.num_samples = 4000
        self.num_images = 12

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): 
        # This function is called at the end of each training batch
        if batch_idx % self.log_steps == 0:

            with torch.no_grad():
                X, X_origin, X_end = batch
                
                Y = pl_module.get_target(X, X_origin, X_end)
                Y = Y[0]

                x_sweeps, sweeps_tags = pl_module.volume_sampling(X, X_origin, X_end)

                Nsweeps = x_sweeps.shape[1]
                
                x = []
                x_v = []

                for n in range(Nsweeps):
                    x_sweeps_n = x_sweeps[:, n, :, :, :, :]
                    sweeps_tags_n = sweeps_tags[:, n]

                    x_sweeps_n, x_sweeps_n_v = pl_module.encode(x_sweeps_n, sweeps_tags_n)
                    
                    x.append(x_sweeps_n)
                    x_v.append(x_sweeps_n_v)

                x = torch.cat(x, dim=1)
                x_v = torch.cat(x_v, dim=1)

                # z, unpooling_idx = pl_module.encode_v(x, x_v)
                # x_hat = pl_module.decoder_v(z, unpooling_idx)
                x_hat = pl_module.encode_v(x, x_v)

                
                fig = self.plot_pointclouds_v4(Y.cpu().numpy(), x_hat[0].cpu().numpy())
                trainer.logger.experiment["images/simulated"].upload(fig)

    def plot_pointclouds_v4(self, Y, X_hat):

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
        )
        
        fig.add_trace(
            go.Scatter3d(x=Y[:,0], y=Y[:,1], z=Y[:,2], mode='markers', marker=dict(
                size=2,
                color=Y[:,2],                # set color to an array/list of desired values
                colorscale='jet',   # choose a colorscale
                opacity=0.8
            )),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter3d(x=X_hat[:,0], y=X_hat[:,1], z=X_hat[:,2], mode='markers', marker=dict(
                size=2,
                color=X_hat[:,2],                # set color to an array/list of desired values
                colorscale='jet',   # choose a colorscale
                opacity=0.8,                
            )),
            row=1, col=2
        )
        # Update the layout if necessary
        fig.update_layout(height=1200, width=1200, title_text="Side-by-Side 3D Scatter Plots")

        return fig

    def plot_pointclouds_v3(self, x_vs, Y, X_hat):

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}], [{'type': 'scatter3d'}, {}]]
        )
        
        fig.add_trace(
            go.Scatter3d(x=Y[:,0], y=Y[:,1], z=Y[:,2], mode='markers', marker=dict(
                size=2,
                color=Y[:,2],                # set color to an array/list of desired values
                colorscale='jet',   # choose a colorscale
                opacity=0.8
            )),
            row=1, col=1
        )

        x_v, x_s = x_vs[0]
        x_v = x_v[0].cpu().numpy()
        x_s = x_s[0].squeeze(1).cpu().numpy()
        
        fig.add_trace(
            go.Scatter3d(x=x_v[:,0], y=x_v[:,1], z=x_v[:,2], text=["%.2f" % s for s in x_s], mode='markers', marker=dict(
                size=2,
                color=x_s,                # set color to an array/list of desired values
                colorscale='jet',   # choose a colorscale
                opacity=0.8,                
            )),
            row=1, col=2
        )
        
        x_v, x_s = x_vs[0]
        x_v = x_v[0].cpu().numpy()
        # x_s = x_s.squeeze(1).cpu().numpy()
        x_s = X_hat.squeeze(1)
        
        threshold = np.percentile(x_s, 95)
        
        x_v = x_v[x_s >= threshold]
        x_s = x_s[x_s >= threshold]
        
        fig.add_trace(
            go.Scatter3d(x=x_v[:,0], y=x_v[:,1], z=x_v[:,2], text=["%.2f" % s for s in x_s], mode='markers', marker=dict(
                size=2,
                color=x_s,                # set color to an array/list of desired values
                colorscale='jet',   # choose a colorscale
                opacity=0.8,                
            )),
            row=2, col=1
        )
        # Update the layout if necessary
        fig.update_layout(height=1200, width=1200, title_text="Side-by-Side 3D Scatter Plots")

        return fig
    
    def plot_pointclouds_v2(self, x_vs, Y, X_hat):

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}], [{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
        )
        
        fig.add_trace(
            go.Scatter3d(x=Y[:,0], y=Y[:,1], z=Y[:,2], mode='markers', marker=dict(
                size=2,
                color=Y[:,2],                # set color to an array/list of desired values
                colorscale='jet',   # choose a colorscale
                opacity=0.8
            )),
            row=1, col=1
        )

        x_v, x_s = x_vs[0]
        x_v = x_v[0].cpu().numpy()
        x_s = x_s[0].squeeze(1).cpu().numpy()
        
        fig.add_trace(
            go.Scatter3d(x=x_v[:,0], y=x_v[:,1], z=x_v[:,2], mode='markers', marker=dict(
                size=2,
                color=x_s,                # set color to an array/list of desired values
                colorscale='jet',   # choose a colorscale
                opacity=0.8,                
            )),
            row=1, col=2
        )
        
        x_v, x_s = x_vs[1]
        x_v = x_v[0].cpu().numpy()
        x_s = x_s[0].squeeze(1).cpu().numpy()
        
        fig.add_trace(
            go.Scatter3d(x=x_v[:,0], y=x_v[:,1], z=x_v[:,2], mode='markers', marker=dict(
                size=2,
                color=x_s,                # set color to an array/list of desired values
                colorscale='jet',   # choose a colorscale
                opacity=0.8,                
            )),
            row=2, col=1
        )
        
        x_v, x_s = x_vs[2]

        x_v = x_v[0].cpu().numpy()
        x_s = X_hat.squeeze()
        
        fig.add_trace(
            go.Scatter3d(x=x_v[:,0], y=x_v[:,1], z=x_v[:,2], mode='markers', marker=dict(
                size=2,
                color=x_s,                # set color to an array/list of desired values
                colorscale='jet',   # choose a colorscale
                opacity=0.8,
            )),
            row=2, col=2
        )


        # Update the layout if necessary
        fig.update_layout(height=1200, width=1200, title_text="Side-by-Side 3D Scatter Plots")

        return fig

    
    def plot_pointclouds(self, V_sweeps, F_sweeps, Y, X_hat):
    

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}], [{'type': 'scatter3d'}, {}]]
        )
        
        fig.add_trace(
            go.Scatter3d(x=Y[:,0], y=Y[:,1], z=Y[:,2], mode='markers', marker=dict(
                size=2,
                color=Y[:,2],                # set color to an array/list of desired values
                colorscale='jet',   # choose a colorscale
                opacity=0.8
            )),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter3d(x=X_hat[:,0], y=X_hat[:,1], z=X_hat[:,2], mode='markers', marker=dict(
                size=2,
                color=X_hat[:,2],                # set color to an array/list of desired values
                colorscale='jet',   # choose a colorscale
                opacity=0.8
            )),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter3d(x=V_sweeps[:,0], y=V_sweeps[:,1], z=V_sweeps[:,2], mode='markers', marker=dict(
                size=2,
                color=F_sweeps,                # set color to an array/list of desired values
                colorscale='gray',   # choose a colorscale
                opacity=0.8
            )),
            row=2, col=1
        )

        


        # Update the layout if necessary
        fig.update_layout(height=1200, width=1200, title_text="Side-by-Side 3D Scatter Plots")

        return fig
    
    def sweep_figure(self, image_data1):
        fig = make_subplots(rows=1, cols=1, subplot_titles=('Image 1', 'Image 2'))

        # Add initial frames for both images with shared coloraxis        
        fig.add_trace(go.Heatmap(z=image_data1[0], coloraxis="coloraxis"), row=1, col=1)

        # Create frames for the animation
        frames = []
        for k in range(image_data1.shape[0]):
            frame = go.Frame(data=[                
                go.Heatmap(z=image_data1[k], coloraxis="coloraxis")
            ], name=str(k))
            frames.append(frame)

        # Add frames to the figure
        fig.frames = frames

        # Calculate the aspect ratio
        height, width = image_data1[0].shape[:2]
        aspect_ratio = height / width

        # Determine global min and max values for consistent color scale
        # vmin = min(image_data1.min(), image_data2.min())
        # vmax = max(image_data1.max(), image_data2.max())
        vmin = image_data1.min()
        vmax = image_data1.max()

        # Update layout with animation settings and fixed aspect ratio
        fig.update_layout(
            autosize=False,
            width=600,  # Adjust width as needed
            height=600,  # Adjust height according to aspect ratio
            coloraxis={"colorscale": "gray",
                    "cmin": vmin,  # Set global min value for color scale
                        "cmax": vmax},   # Set global max value for color scale},  # Set colorscale for the shared coloraxis
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True},
                                        "fromcurrent": True, "mode": "immediate"}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "steps": [
                    {
                        "args": [[str(k)], {"frame": {"duration": 300, "redraw": True},
                                            "mode": "immediate"}],
                        "label": str(k),
                        "method": "animate"
                    } for k in range(image_data1.shape[0])
                ],
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Frame:",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"}
            }]
        )
        return fig
    
class UNetReconstructionLoggerNeptune(Callback):
    # This callback logs images for visualization during training, with the ability to log images to the Neptune logging system for easy monitoring and analysis
    def __init__(self, num_surf=1, log_steps=10):
        self.log_steps = log_steps
        self.num_surf = num_surf
        self.num_samples = 1000
        self.num_images = 12

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): 
        # This function is called at the end of each training batch
        if batch_idx % self.log_steps == 0:

            with torch.no_grad():
                X, X_origin, X_end = batch
                
                X_fov, X_fov_mask = pl_module.get_sweeps(X, X_origin, X_end)

                Y_fov = pl_module.get_target(X, X_origin, X_end, X_fov, X_fov_mask)
                
                X_hat = pl_module(X_fov)

                X_vs = None
                X_vw = None
                if isinstance(X_hat, tuple):
                    X_hat, X_vs, X_vw = X_hat

                X_hat = torch.argmax(X_hat, dim=1, keepdim=True).float()
                fig = self.create_figure(X_fov[1][0].cpu().numpy(), Y_fov[1][0].cpu().numpy(), X_hat[1][0].cpu().numpy())


                # X_fov = X_fov[0].reshape(-1)
                # Y_fov = Y_fov[0].reshape(-1)
                # X_hat = X_hat[0].reshape(-1)


                # V_fov = pl_module.vs.fov_physical().reshape(-1, 3).to(pl_module.device)
                
                # V_filtered = V_fov[X_fov > pl_module.hparams.threshold]
                # F_filtered = X_fov[X_fov > pl_module.hparams.threshold]                
                
                # random_indices = torch.randperm(V_filtered.size(0))[:self.num_samples]
                # V_filtered = V_filtered[random_indices, :]
                # F_filtered = F_filtered[random_indices]


                # V_Y_fov = V_fov[Y_fov == 1]
                # random_indices = torch.randperm(V_Y_fov.size(0))[:self.num_samples]
                # V_Y_fov = V_Y_fov[random_indices, :]

                # X_hat = torch.sigmoid(X_hat)
                # V_X_hat = V_fov[X_hat > 0.9]
                # random_indices = torch.randperm(V_X_hat.size(0))[:self.num_samples]
                # V_X_hat = V_X_hat[random_indices, :]

                # fig = self.plot_pointclouds(V_filtered.cpu().numpy(), F_filtered.cpu().numpy(), V_Y_fov.cpu().numpy(), V_X_hat.cpu().numpy())
                trainer.logger.experiment["images/simulated"].upload(fig)

    
    def plot_pointclouds(self, V_sweeps, F_sweeps, Y, X_hat):
    

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scatter3d'}, {}], [{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
        )

        # First scatter plot
        fig.add_trace(
            go.Scatter3d(x=V_sweeps[:,0], y=V_sweeps[:,1], z=V_sweeps[:,2], mode='markers', marker=dict(
                size=2,
                color=F_sweeps,                # set color to an array/list of desired values
                colorscale='gray',   # choose a colorscale
                opacity=0.8
            )),
            row=1, col=1
        )

        # Second scatter plot
        fig.add_trace(
            go.Scatter3d(x=Y[:,0], y=Y[:,1], z=Y[:,2], mode='markers', marker=dict(
                size=2,
                color=Y[:,2],                # set color to an array/list of desired values
                colorscale='jet',   # choose a colorscale
                opacity=0.8
            )),
            row=2, col=1
        )

        # Second scatter plot
        fig.add_trace(
            go.Scatter3d(x=X_hat[:,0], y=X_hat[:,1], z=X_hat[:,2], mode='markers', marker=dict(
                size=2,
                color=X_hat[:,2],                # set color to an array/list of desired values
                colorscale='jet',   # choose a colorscale
                opacity=0.8
            )),
            row=2, col=2
        )


        # Update the layout if necessary
        fig.update_layout(height=1200, width=1200, title_text="Side-by-Side 3D Scatter Plots")

        return fig
    
    def create_figure(self, image_data1, image_data2, image_data3):
        fig = make_subplots(rows=1, cols=3, subplot_titles=('Sweep', 'GT', 'Reconstruction'))

        # Add initial frames for both images with shared coloraxis
        fig.add_trace(go.Heatmap(z=image_data1[0], colorscale = 'gray'), row=1, col=1)
        fig.add_trace(go.Heatmap(z=image_data2[0], coloraxis="coloraxis"), row=1, col=2)
        fig.add_trace(go.Heatmap(z=image_data3[0], coloraxis="coloraxis"), row=1, col=3)

        # Create frames for the animation
        frames = []
        for k in range(image_data1.shape[0]):
            frame = go.Frame(data=[
                go.Heatmap(z=image_data1[k], colorscale = 'gray'),
                go.Heatmap(z=image_data2[k], coloraxis="coloraxis"),
                go.Heatmap(z=image_data3[k], coloraxis="coloraxis")
            ], name=str(k))
            frames.append(frame)

        # Add frames to the figure
        fig.frames = frames

        # Calculate the aspect ratio
        height, width = image_data1[0].shape[:2]
        aspect_ratio = height / width

        # Determine global min and max values for consistent color scale
        # vmin = min(image_data1.min(), image_data2.min())
        # vmax = max(image_data1.max(), image_data2.max())
        vmin = image_data2.min()
        vmax = image_data2.max()

        # Update layout with animation settings and fixed aspect ratio
        fig.update_layout(
            autosize=False,
            width=1200,  # Adjust width as needed
            height=600,  # Adjust height according to aspect ratio
            coloraxis={"colorscale": "jet",
                    "cmin": vmin,  # Set global min value for color scale
                        "cmax": vmax},   # Set global max value for color scale},  # Set colorscale for the shared coloraxis
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True},
                                        "fromcurrent": True, "mode": "immediate"}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "steps": [
                    {
                        "args": [[str(k)], {"frame": {"duration": 300, "redraw": True},
                                            "mode": "immediate"}],
                        "label": str(k),
                        "method": "animate"
                    } for k in range(image_data1.shape[0])
                ],
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Frame:",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"}
            }]
        )
        return fig
    
class UNetReconstruction2DLoggerNeptune(Callback):
    # This callback logs images for visualization during training, with the ability to log images to the Neptune logging system for easy monitoring and analysis
    def __init__(self, num_surf=1, log_steps=10):
        self.log_steps = log_steps
        self.num_surf = num_surf
        self.num_samples = 1000
        self.num_images = 16
        self.start = 64

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): 
        # This function is called at the end of each training batch
        if batch_idx % self.log_steps == 0:

            with torch.no_grad():
                X, X_origin, X_end = batch
                
                X_sweep_simu, X_sweep = pl_module.get_sweeps(X, X_origin, X_end)
                
                X_hat = pl_module(X_sweep_simu)

                X_hat = torch.argmax(X_hat, dim=1, keepdim=True).float()
                fig = self.create_figure(X_sweep_simu[0][0][self.start: self.start + self.num_images].cpu().numpy(), X_sweep[0][0][self.start:self.start + self.num_images].cpu().numpy(), X_hat[0][0][self.start: self.start + self.num_images].cpu().numpy())


                # X_fov = X_fov[0].reshape(-1)
                # Y_fov = Y_fov[0].reshape(-1)
                # X_hat = X_hat[0].reshape(-1)


                # V_fov = pl_module.vs.fov_physical().reshape(-1, 3).to(pl_module.device)
                
                # V_filtered = V_fov[X_fov > pl_module.hparams.threshold]
                # F_filtered = X_fov[X_fov > pl_module.hparams.threshold]                
                
                # random_indices = torch.randperm(V_filtered.size(0))[:self.num_samples]
                # V_filtered = V_filtered[random_indices, :]
                # F_filtered = F_filtered[random_indices]


                # V_Y_fov = V_fov[Y_fov == 1]
                # random_indices = torch.randperm(V_Y_fov.size(0))[:self.num_samples]
                # V_Y_fov = V_Y_fov[random_indices, :]

                # X_hat = torch.sigmoid(X_hat)
                # V_X_hat = V_fov[X_hat > 0.9]
                # random_indices = torch.randperm(V_X_hat.size(0))[:self.num_samples]
                # V_X_hat = V_X_hat[random_indices, :]

                # fig = self.plot_pointclouds(V_filtered.cpu().numpy(), F_filtered.cpu().numpy(), V_Y_fov.cpu().numpy(), V_X_hat.cpu().numpy())
                trainer.logger.experiment["images/simulated"].upload(fig)

    
    def plot_pointclouds(self, V_sweeps, F_sweeps, Y, X_hat):
    

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scatter3d'}, {}], [{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
        )

        # First scatter plot
        fig.add_trace(
            go.Scatter3d(x=V_sweeps[:,0], y=V_sweeps[:,1], z=V_sweeps[:,2], mode='markers', marker=dict(
                size=2,
                color=F_sweeps,                # set color to an array/list of desired values
                colorscale='gray',   # choose a colorscale
                opacity=0.8
            )),
            row=1, col=1
        )

        # Second scatter plot
        fig.add_trace(
            go.Scatter3d(x=Y[:,0], y=Y[:,1], z=Y[:,2], mode='markers', marker=dict(
                size=2,
                color=Y[:,2],                # set color to an array/list of desired values
                colorscale='jet',   # choose a colorscale
                opacity=0.8
            )),
            row=2, col=1
        )

        # Second scatter plot
        fig.add_trace(
            go.Scatter3d(x=X_hat[:,0], y=X_hat[:,1], z=X_hat[:,2], mode='markers', marker=dict(
                size=2,
                color=X_hat[:,2],                # set color to an array/list of desired values
                colorscale='jet',   # choose a colorscale
                opacity=0.8
            )),
            row=2, col=2
        )


        # Update the layout if necessary
        fig.update_layout(height=1200, width=1200, title_text="Side-by-Side 3D Scatter Plots")

        return fig
    
    def create_figure(self, image_data1, image_data2, image_data3):
        fig = make_subplots(rows=1, cols=3, subplot_titles=('Sweep', 'GT', 'Reconstruction'))

        # Add initial frames for both images with shared coloraxis
        fig.add_trace(go.Heatmap(z=image_data1[0], colorscale = 'gray'), row=1, col=1)
        fig.add_trace(go.Heatmap(z=image_data2[0], coloraxis="coloraxis"), row=1, col=2)
        fig.add_trace(go.Heatmap(z=image_data3[0], coloraxis="coloraxis"), row=1, col=3)

        # Create frames for the animation
        frames = []
        for k in range(image_data1.shape[0]):
            frame = go.Frame(data=[
                go.Heatmap(z=image_data1[k], colorscale = 'gray'),
                go.Heatmap(z=image_data2[k], coloraxis="coloraxis"),
                go.Heatmap(z=image_data3[k], coloraxis="coloraxis")
            ], name=str(k))
            frames.append(frame)

        # Add frames to the figure
        fig.frames = frames

        # Calculate the aspect ratio
        height, width = image_data1[0].shape[:2]
        aspect_ratio = height / width

        # Determine global min and max values for consistent color scale
        # vmin = min(image_data1.min(), image_data2.min())
        # vmax = max(image_data1.max(), image_data2.max())
        vmin = image_data2.min()
        vmax = image_data2.max()

        # Update layout with animation settings and fixed aspect ratio
        fig.update_layout(
            autosize=False,
            width=1200,  # Adjust width as needed
            height=600,  # Adjust height according to aspect ratio
            coloraxis={"colorscale": "jet",
                    "cmin": vmin,  # Set global min value for color scale
                        "cmax": vmax},   # Set global max value for color scale},  # Set colorscale for the shared coloraxis
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True},
                                        "fromcurrent": True, "mode": "immediate"}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "steps": [
                    {
                        "args": [[str(k)], {"frame": {"duration": 300, "redraw": True},
                                            "mode": "immediate"}],
                        "label": str(k),
                        "method": "animate"
                    } for k in range(image_data1.shape[0])
                ],
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Frame:",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"}
            }]
        )
        return fig
    
class FluidLogger(Callback):
    # This callback logs images for visualization during training, with the ability to log images to the Neptune logging system for easy monitoring and analysis
    def __init__(self, num_surf=1, log_steps=10):
        self.log_steps = log_steps
        self.num_surf = num_surf
        self.num_samples = 10000

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): 
        # This function is called at the end of each training batch
        if batch_idx % self.log_steps == 0:

            with torch.no_grad():
                X_d = batch
                X_v, X_f = X_d["img"]
                Y_v, Y = X_d["seg"]
                
                x = pl_module(X_v, X_f)

                random_indices = torch.randperm(X_v.size(1))[:self.num_samples]
                X_v = X_v[0, random_indices, :]
                X_f = X_f[0, random_indices, 0]
                Y_v = Y_v[0, random_indices, :]
                Y = Y[0, random_indices, :]

                x = torch.argmax(x[0, random_indices, :], dim=1).float()
                # x = x[0, random_indices].squeeze()
                
                fig = self.plot_pointclouds(X_v.cpu().numpy(), X_f.squeeze().cpu().numpy(), x.cpu().numpy(), Y_v.cpu().numpy(), Y.squeeze(1).cpu().numpy())
                trainer.logger.experiment["images/seg"].upload(fig)


    def plot_pointclouds(self, X_v, X_f, x, Y_v, Y):

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scatter3d'}, {}], [{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
        )

        fig.add_trace(
            go.Scatter3d(x=X_v[:,0], y=X_v[:,1], z=X_v[:,2], text=["%.2f" % s for s in X_f], mode='markers', marker=dict(
                size=2,
                color=X_f,                # set color to an array/list of desired values
                colorscale='jet',   # choose a colorscale
                opacity=0.8,                
            )),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter3d(x=Y_v[:,0], y=Y_v[:,1], z=Y_v[:,2], text=["%.2f" % s for s in Y], mode='markers', marker=dict(
                size=2,
                color=Y,                # set color to an array/list of desired values
                colorscale='jet',   # choose a colorscale
                opacity=0.8
            )),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter3d(x=X_v[:,0], y=X_v[:,1], z=X_v[:,2], text=["%.2f" % s for s in x], mode='markers', marker=dict(
                size=2,
                color=x,                # set color to an array/list of desired values
                colorscale='jet',   # choose a colorscale
                opacity=0.8,                
            )),
            row=2, col=2
        )

        # Update the layout if necessary
        fig.update_layout(height=1200, width=1200, title_text="Side-by-Side 3D Scatter Plots")

        return fig




class USGAPCLoggerNeptune(Callback):
    # This callback logs images for visualization during training, with the ability to log images to the Neptune logging system for easy monitoring and analysis
    def __init__(self, num_surf=1, log_steps=10):
        self.log_steps = log_steps
        self.num_surf = num_surf
        self.num_images = 12

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): 
        # This function is called at the end of each training batch
        if batch_idx % self.log_steps == 0:

            with torch.no_grad():
                x_d = batch
        
                x = []
                x_v = []

                for k in range(pl_module.hparams.max_sweeps):
                    x_img = x_d[k]
                    tags = x_d["tag"][:,k]

                    x_, x_v_, x_e = pl_module.encode(x_img, tags)

                    x.append(x_)
                    x_v.append(x_v_)

                x = torch.cat(x, dim=1)
                x_v = torch.cat(x_v, dim=1)
                
                # fig = self.plot_pointclouds(V_sweeps[0].cpu().numpy(), F_sweeps[0].squeeze().cpu().numpy(), Y[0].cpu().numpy(), X_hat[0].cpu().numpy())
                fig = self.plot_pointclouds(x_v[0].cpu().numpy())
                trainer.logger.experiment["images/simulated"].upload(fig)

                fig = self.sweep_figure(x_e[0].permute(1, 2, 3, 0).cpu().numpy())
                trainer.logger.experiment["images/encoded"].upload(fig)


    def plot_pointclouds(self, x_v):

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}], [{'type': 'scatter3d'}, {}]]
        )
        
        fig.add_trace(
            go.Scatter3d(x=x_v[:,0], y=x_v[:,1], z=x_v[:,2], mode='markers', marker=dict(
                size=2,
                color=x_v[:,2],                # set color to an array/list of desired values
                colorscale='jet',   # choose a colorscale
                opacity=0.8
            )),
            row=1, col=1
        )
        
        fig.update_layout(height=1200, width=1200, title_text="Side-by-Side 3D Scatter Plots")

        return fig
    
    def sweep_figure(self, image_data1):

        # Normalize image_data1 between 0-255
        
        D, H, W, C = image_data1.shape

        image_data1 = (image_data1 - image_data1.min()) / (image_data1.max() - image_data1.min()) * 255

        fig = make_subplots(rows=1, cols=1, subplot_titles=('Image 1', 'Image 2'))

        # Add initial frames for both images with shared coloraxis        
        fig.add_trace(go.Image(z=image_data1[0]), row=1, col=1)

        # Create frames for the animation
        frames = []
        for k in range(D):
            frame = go.Frame(data=[                
                go.Image(z=image_data1[k])
            ], name=str(k))
            frames.append(frame)

        # Add frames to the figure
        fig.frames = frames

        # Calculate the aspect ratio
        aspect_ratio = H / W

        # Determine global min and max values for consistent color scale
        # vmin = min(image_data1.min(), image_data2.min())
        # vmax = max(image_data1.max(), image_data2.max())
        vmin = image_data1.min()
        vmax = image_data1.max()

        # Update layout with animation settings and fixed aspect ratio
        fig.update_layout(
            autosize=False,
            width=600,  # Adjust width as needed
            height=600,  # Adjust height according to aspect ratio
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True},
                                        "fromcurrent": True, "mode": "immediate"}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "steps": [
                    {
                        "args": [[str(k)], {"frame": {"duration": 300, "redraw": True},
                                            "mode": "immediate"}],
                        "label": str(k),
                        "method": "animate"
                    } for k in range(D)
                ],
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Frame:",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"}
            }]
        )
        return fig


class USDDPMPCLoggerNeptune(Callback):
    # This callback logs images for visualization during training, with the ability to log images to the Neptune logging system for easy monitoring and analysis
    def __init__(self, num_surf=5, log_steps=10, num_steps=5):
        self.log_steps = log_steps
        self.num_surf = num_surf
        self.num_steps = num_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): 
        # This function is called at the end of each training batch
        if pl_module.global_step % self.log_steps == 0:
            
            pl_module.eval()
            with torch.no_grad():

                X, X_origin, X_end, X_PC = batch
                X = X[0:1]
                X_origin = X_origin[0:1] 
                X_end = X_end[0:1]
                X_PC = X_PC[0:1]

                x_sweeps, sweeps_tags = pl_module.volume_sampling(X, X_origin, X_end)

                # x_sweeps shape is B, N, C, T, H, W. N for number of sweeps ex. torch.Size([2, 2, 1, 200, 256, 256]) 
                # tags shape torch.Size([2, 2])

                batch_size = x_sweeps.shape[0]
                Nsweeps = x_sweeps.shape[1] # Number of sweeps -> T
                
                z = []
                x_v = []

                for n in range(Nsweeps):
                    x_sweeps_n = x_sweeps[:, n, :, :, :, :] # [BS, C, T, H, W]
                    sweeps_tags_n = sweeps_tags[:, n]

                    z_mu, z_sigma = pl_module.encode(x_sweeps_n)
                    z_ = z_mu

                    z_ = pl_module.attn_chunk(z_) # [BS, self.hparams.latent_channels, self.hparams.n_chunks, 64. 64]

                    z_ = z_.permute(0, 2, 3, 4, 1).reshape(batch_size, pl_module.hparams.n_chunks, -1) # [BS, self.hparams.n_chunks, 64*64*self.hparams.latent_channels]

                    z.append(z_.unsqueeze(1))

                z = torch.cat(z, dim=1) # [BS, N, self.hparams.n_chunks, 64*64*self.hparams.latent_channels]

                z = pl_module.proj(z) # [BS, N, elf.hparams.n_chunks, 1280]

                # We don't need to do the trick of using the buffer for the positional encoding here, ALL the sweeps are present in validation
                z = pl_module.p_encoding(z)
                z = z.view(batch_size, -1, pl_module.hparams.embed_dim).contiguous()

                
                fig = self.plot_diffusion(X_PC[0:self.num_surf].cpu().numpy())
                trainer.logger.experiment["images/batch"].upload(fig)
                
                pc, intermediates = pl_module.sample(intermediate_steps=self.num_steps, z=z)
                
                fig = self.plot_diffusion(torch.cat(intermediates, dim=0).cpu().numpy())
                trainer.logger.experiment["images/intermediates"].upload(fig)

                

    def plot_diffusion(self, X):
        num_surf = len(X)
        specs_r = [{'type': 'scatter3d'} for _ in range(num_surf)]

        fig = make_subplots(
            rows=1, cols=num_surf,
            specs=[specs_r]
        )

        for idx, x in zip(range(num_surf), X):
            # First scatter plot
            fig.add_trace(
                go.Scatter3d(x=x[:,0], y=x[:,1], z=x[:,2], mode='markers', marker=dict(
                    size=2,
                    color=x[:,2],                # set color to an array/list of desired values
                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.8
                )),
                row=1, col=idx+1
            )

        return fig


    def plot_pointclouds(self, X, X_noised, X_hat):

        num_surf = len(X)
        specs_r = [{'type': 'scatter3d'} for _ in range(num_surf)]

        fig = make_subplots(
            rows=3, cols=num_surf,
            specs=[specs_r, specs_r, specs_r]
        )

        for idx, x, x_noised, x_hat in zip(range(num_surf), X, X_noised, X_hat):
            # First scatter plot
            fig.add_trace(
                go.Scatter3d(x=x[:,0], y=x[:,1], z=x[:,2], mode='markers', marker=dict(
                    size=2,
                    color=x[:,2],                # set color to an array/list of desired values
                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.8
                )),
                row=1, col=idx+1
            )

            # Second scatter plot
            fig.add_trace(
                go.Scatter3d(x=x_noised[:,0], y=x_noised[:,1], z=x_noised[:,2], mode='markers', marker=dict(
                    size=2,
                    color=x_noised[:,2],                # set color to an array/list of desired values
                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.8
                )),
                row=2, col=idx+1
            )

            # Third scatter plot
            fig.add_trace(
                go.Scatter3d(x=x_hat[:,0], y=x_hat[:,1], z=x_hat[:,2], mode='markers', marker=dict(
                    size=2,
                    color=x_hat[:,2],                # set color to an array/list of desired values
                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.8
                )),
                row=3, col=idx+1
            )

        # Update the layout if necessary
        fig.update_layout(height=900, width=1600, title_text="Side-by-Side 3D Scatter Plots")

        return fig
    

class USBabyFrameLoggerNeptune(Callback):
    # This callback logs images for visualization during training, with the ability to log images to the Neptune logging system for easy monitoring and analysis
    def __init__(self, num_surf=5, log_steps=10, num_steps=5):
        self.log_steps = log_steps
        self.num_surf = num_surf
        self.num_steps = num_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): 
        # This function is called at the end of each training batch
        if pl_module.global_step % self.log_steps == 0:
            
            pl_module.eval()
            with torch.no_grad():

                X, X_origin, X_end, X_PC = batch
                X = X[0:1]
                X_origin = X_origin[0:1] 
                X_end = X_end[0:1]
                X_PC = X_PC[0:1]

                x_sweeps, sweeps_tags = pl_module.volume_sampling(X, X_origin, X_end)

                x_hat = pl_module(x_sweeps, sweeps_tags)

                y, points = pl_module.compute_orthogonal_frame(X_PC)        
                
                fig = self.plot_diffusion(X_PC[0:self.num_surf].cpu().numpy(), points[0:self.num_surf].cpu().numpy(), y[0:self.num_surf].cpu().numpy(), x_hat[0:self.num_surf].cpu().numpy())
                trainer.logger.experiment["images/batch"].upload(fig)
                

    def plot_diffusion(self, X, P, frame, frame_pred):
        num_surf = len(X)
        specs_r = [{'type': 'scatter3d'} for _ in range(num_surf)]

        fig = make_subplots(
            rows=1, cols=num_surf,
            specs=[specs_r]
        )

        for idx, x in zip(range(num_surf), X):
            # First scatter plot
            fig.add_trace(
                go.Scatter3d(x=x[:,0], y=x[:,1], z=x[:,2], mode='markers', marker=dict(
                    size=2,
                    color=x[:,2],                # set color to an array/list of desired values
                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.8
                )),
                row=1, col=idx+1
            )

        if P is not None:
            for idx, p in zip(range(num_surf), P):
                fig.add_trace(
                    go.Scatter3d(x=p[:,0], y=p[:,1], z=p[:,2], mode='markers', marker=dict(
                        size=4,
                        color='red',                # set color to an array/list of desired values
                        opacity=1,
                    )),
                    row=1, col=idx+1
                )

        if frame is not None:
            for idx, f in zip(range(num_surf), frame):

                origin = P[idx,0]

                for i in range(3):
                    fig.add_trace(go.Scatter3d(
                        x=[origin[0], origin[0] + f[i, 0]],
                        y=[origin[1], origin[1] + f[i, 1]],
                        z=[origin[2], origin[2] + f[i, 2]],
                        mode='lines',
                        line=dict(color='cyan', width=8),
                    ), 
                    row=1, col=idx+1)

        if frame_pred is not None:
            for idx, f in zip(range(num_surf), frame_pred):

                origin = P[idx,0]     

                for i in range(3):
                    fig.add_trace(go.Scatter3d(
                        x=[origin[0], origin[0] + f[i, 0]],
                        y=[origin[1], origin[1] + f[i, 1]],
                        z=[origin[2], origin[2] + f[i, 2]],
                        mode='lines',
                        line=dict(color='magenta', width=8),
                    ), 
                    row=1, col=idx+1)

        return fig

class USSegLoggerNeptune(Callback):
    # This callback logs images for visualization during training, with the ability to log images to the Neptune logging system for easy monitoring and analysis
    def __init__(self, num_images=5, log_steps=10, num_steps=5):
        self.log_steps = log_steps
        self.num_images = num_images
        self.num_steps = num_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): 
        # This function is called at the end of each training batch
        if pl_module.global_step % self.log_steps == 0:
            
            pl_module.eval()
            with torch.no_grad():

                X, X_origin, X_end, X_PC = batch
                X = X[0:1]
                X_origin = X_origin[0:1] 
                X_end = X_end[0:1]
                X_PC = X_PC[0:1]

                x, y, sweeps_tags = pl_module.volume_sampling(X, X_origin, X_end)
                x = x.squeeze(1)
                x_hat = pl_module(x)
                x_hat = torch.argmax(x_hat, dim=1, keepdim=True)

                r_idx = torch.randperm(x.shape[2])[:self.num_images]
                
                x = x[:,:,r_idx, :, :]
                x_hat = x_hat[:,:,r_idx, :, :]
                y = y[:,:,r_idx, :, :]

                x = x[0][0].cpu().numpy()
                x_hat = x_hat[0][0].cpu().numpy()
                y = y[0][0].cpu().numpy()

                fig = self.plot_seq(x, x_hat, y)
                trainer.logger.experiment["images/batch"].upload(fig)
                

    def plot_seq(self, x, x_hat, y):
        
        T, H, W = x.shape
        # Create subplot figure with 1 row and 2 columns
        fig = make_subplots(rows=1, cols=3, subplot_titles=["x", "x_hat", "y"])

        # Initial frame
        fig.add_trace(go.Heatmap(z=x[0], colorscale = 'gray'), row=1, col=1)
        fig.add_trace(go.Heatmap(z=x_hat[0], coloraxis = 'coloraxis'), row=1, col=2)
        fig.add_trace(go.Heatmap(z=y[0], coloraxis = 'coloraxis'), row=1, col=3)

        # Add animation frames
        frames = []
        for i in range(T):
            frames.append(go.Frame(
                data=[
                    go.Heatmap(z=x[i], colorscale = 'gray'),
                    go.Heatmap(z=x_hat[i], coloraxis = 'coloraxis'),
                    go.Heatmap(z=y[i], coloraxis = 'coloraxis')
                ],
                name=str(i)
            ))

        fig.frames = frames

        # Update layout with animation settings and fixed aspect ratio
        fig.update_layout(
            autosize=False,
            width=1800,  # Adjust width as needed
            height=600,  # Adjust height according to aspect ratio
            coloraxis={"colorscale": "jet",
                    "cmin": 0,  # Set global min value for color scale
                    "cmax": 11},   # Set global max value for color scale},  # Set colorscale for the shared coloraxis
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True},
                                        "fromcurrent": True, "mode": "immediate"}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "steps": [
                    {
                        "args": [[str(k)], {"frame": {"duration": 300, "redraw": True},
                                            "mode": "immediate"}],
                        "label": str(k),
                        "method": "animate"
                    } for k in range(x.shape[0])
                ],
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Frame:",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"}
            }]
        )
        return fig
    

class Cut3DLogger(Callback):
    def __init__(self, num_images=1, log_steps=100, *args, **kwargs):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():

                Y = batch
                Y = pl_module.resize_t(Y)

                max_num_image = min(Y.shape[0], self.num_images)
                Y = Y[0:max_num_image]

                X, tags = pl_module.volume_sampling(pl_module.diffusor_t, pl_module.diffusor_origin, pl_module.diffusor_end)
                X = X[0][0:max_num_image]
                X = pl_module.resize_t(X)

                Y_fake = pl_module(X)
                
                X = torch.clip(X, min=0.0, max=1.0)
                Y_fake = torch.clip(Y_fake, min=0.0, max=1.0)

                X = X[0][0].unsqueeze(1)
                Y_fake = Y_fake[0][0].unsqueeze(1)
                Y = Y[0][0].unsqueeze(1)

                grid_x = torchvision.utils.make_grid(X, nrow=8)

                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x"].upload(fig)
                plt.close()

                
                grid_y_fake = torchvision.utils.make_grid(Y_fake, nrow=8)
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_y_fake.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/y_fake"].upload(fig)
                plt.close()

                grid_y = torchvision.utils.make_grid(Y, nrow=8)
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_y.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/y"].upload(fig)
                plt.close()