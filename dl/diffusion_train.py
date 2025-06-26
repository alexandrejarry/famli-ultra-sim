
import argparse

import os
import sys

import torch

import lightning as L
from lightning.pytorch.core import LightningModule, LightningDataModule

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from lightning.pytorch.loggers import NeptuneLogger

from loaders import ultrasound_dataset as usd
from nets import diffusion
from callbacks import logger

def train(args, callbacks):

    deterministic = None
    if args.seed_everything:
        seed_everything(args.seed_everything, workers=True)
        deterministic = True

    DATAMODULE = getattr(usd, args.data_module)

    args_d = vars(args)

    data = DATAMODULE(**args_d)
    
    SAXINETS = getattr(diffusion, args.nn)
    model = SAXINETS(**args_d)
    
    logger_neptune = None
    if args.neptune_tags:
        logger_neptune = NeptuneLogger(
            project='ImageMindAnalytics/us-simu',
            tags=args.neptune_tags,
            api_key=os.environ['NEPTUNE_API_TOKEN'],
            log_model_checkpoints=False
        )
        LOGGER = getattr(logger, args.logger)    
        image_logger = LOGGER(**args_d)
        callbacks.append(image_logger)

    trainer = Trainer(logger=logger_neptune, 
        max_epochs=args.epochs, 
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        strategy=DDPStrategy(find_unused_parameters=args.find_unused_parameters),
        gradient_clip_val=args.gradient_clip_val,
        deterministic=deterministic)
    trainer.fit(model, datamodule=data, ckpt_path=args.model)

def main(args):

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss',
        save_last=bool(args.save_last)
    )
    
    # Early Stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00, 
        patience=args.patience, 
        verbose=True, 
        mode="min"
    )

    train(args, [checkpoint_callback, early_stop_callback])


def get_argparse():
    parser = argparse.ArgumentParser(description='Contrastive Unpaired Image To Image Translation', add_help=False)

    hparams_group = parser.add_argument_group('Hyperparameters')
    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    hparams_group.add_argument('--patience', help='Max number of patience for early stopping', type=int, default=30)
    hparams_group.add_argument('--steps', help='Max number of steps per epoch', type=int, default=-1)
    hparams_group.add_argument('--gradient_clip_val', help='Gradient clipping for the trainer', type=float, default=None)
    hparams_group.add_argument('--seed_everything', help='Seed everything for training', type=int, default=None)
    hparams_group.add_argument('--find_unused_parameters', help='Find unused parameters', type=int, default=0)
    
    
    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--nn', help='Neural network name', type=str, default=None)
    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)
    
    input_group.add_argument('--data_module', help='Data module type', type=str, default=None)

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")
    output_group.add_argument('--use_early_stopping', help='Use early stopping criteria', type=int, default=0)
    output_group.add_argument('--save_last', help='Save last checkpoint as well', type=int, default=0)
    # output_group.add_argument('--monitor', help='Additional metric to monitor to save checkpoints', type=str, default=None)
    
    ##Logger
    logger_group = parser.add_argument_group('Logger')
    logger_group.add_argument('--logger', type=str, help='Logger class name', default=None)
    logger_group.add_argument('--log_every_n_steps', type=int, help='Log every n steps during training', default=10)    
    
    logger_group.add_argument('--neptune_project', type=str, help='Neptune project', default=None)
    logger_group.add_argument('--neptune_tags', type=str, nargs='+', help='Neptune tags', default=None)
    logger_group.add_argument('--neptune_token', type=str, help='Neptune token', default=None)

    return parser


def dynamically_add_args(parser, component_name, registry, arg_adder_name):
    if component_name:
        if not hasattr(registry, component_name):
            raise ValueError(f"{component_name} not found in {registry.__name__}")
        component_class = getattr(registry, component_name)
        if hasattr(component_class, arg_adder_name):
            parser = getattr(component_class, arg_adder_name)(parser)
    return parser

def list_available_models(registry):
    return [
        name for name in dir(registry)
        if not name.startswith("_")
        and isinstance(getattr(registry, name), type)
        and (
            issubclass(getattr(registry, name), LightningModule)
            or issubclass(getattr(registry, name), LightningDataModule)
        )
        and name not in ["LightningModule", "LightningDataModule"]
    ]

if __name__ == '__main__':
    parser = get_argparse()  # this adds --nn, --data_module, --logger, etc.
    initial_args, _ = parser.parse_known_args()  # parse known args to get nn, data_module, logger

    # Now add args based on provided --nn, --data_module, etc.
    parser = dynamically_add_args(parser, initial_args.nn, diffusion, 'add_model_specific_args')
    parser = dynamically_add_args(parser, initial_args.data_module, usd, 'add_data_specific_args')
    parser = dynamically_add_args(parser, initial_args.logger, logger, 'add_logger_specific_args')

    # Inside __main__ block, after initial_args = parser.parse_known_args()
    if initial_args.nn is None:
        print("❌ No --nn specified.")
        print("✅ Available options for --nn:")
        for name in list_available_models(diffusion):
            print(f"  - {name}")
        sys.exit(1)

    # Inside __main__ block, after initial_args = parser.parse_known_args()
    if initial_args.data_module is None:
        print("❌ No --data_module specified.")
        print("✅ Available options for --data_module:")
        for name in list_available_models(usd):
            print(f"  - {name}")
        sys.exit(1)

    # If help flag is used, print full help after dynamic args are added
    if any(arg in sys.argv for arg in ['-h', '--help']):
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    main(args)