# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
os.environ["TORCH_USE_WEIGHTS_ONLY_UNPICKLER"] = "0"

import logging
import os
import pprint
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, List, Optional

import hydra
import pytorch_lightning as pl
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from emg2qwerty import transforms, utils
from emg2qwerty.transforms import Transform

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

log = logging.getLogger(__name__)

# Simple top-level flag to enable/disable WandB
USE_WANDB = False  # Set this to False to disable WandB logging


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    log.info(f"\nConfig:\n{OmegaConf.to_yaml(config)}")
    
    log.info(f"WandB logging is {'enabled' if USE_WANDB else 'disabled'}")

    # Add working dir to PYTHONPATH
    working_dir = get_original_cwd()
    python_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if working_dir not in python_paths:
        python_paths.append(working_dir)
        os.environ["PYTHONPATH"] = os.pathsep.join(python_paths)

    # Seed for determinism. This seeds torch, numpy and python random modules
    # taking global rank into account (for multi-process distributed setting).
    # Additionally, this auto-adds a worker_init_fn to train_dataloader that
    # initializes the seed taking worker_id into account per dataloading worker
    # (see `pl_worker_init_fn()`).
    pl.seed_everything(config.seed, workers=True)

    # Helper to instantiate full paths for dataset sessions
    def _full_session_paths(dataset: ListConfig) -> list[Path]:
        sessions = [session["session"] for session in dataset]
        return [
            Path(config.dataset.root).joinpath(f"{session}.hdf5")
            for session in sessions
        ]

    # Helper to instantiate transforms
    def _build_transform(configs: Sequence[DictConfig]) -> Transform[Any, Any]:
        return transforms.Compose([instantiate(cfg) for cfg in configs])

    # Instantiate LightningModule
    log.info(f"Instantiating LightningModule {config.module}")
    module = instantiate(
        config.module,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        decoder=config.decoder,
        _recursive_=False,
    )
    if config.checkpoint is not None:
        log.info(f"Loading module from checkpoint {config.checkpoint}")
        module = module.load_from_checkpoint(
            config.checkpoint,
            optimizer=config.optimizer,
            lr_scheduler=config.lr_scheduler,
            decoder=config.decoder,
        )

    # Instantiate LightningDataModule
    log.info(f"Instantiating LightningDataModule {config.datamodule}")
    datamodule = instantiate(
        config.datamodule,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_sessions=_full_session_paths(config.dataset.train),
        val_sessions=_full_session_paths(config.dataset.val),
        test_sessions=_full_session_paths(config.dataset.test),
        train_transform=_build_transform(config.transforms.train),
        val_transform=_build_transform(config.transforms.val),
        test_transform=_build_transform(config.transforms.test),
        _convert_="object",
    )

    # Instantiate callbacks
    callback_configs = config.get("callbacks", [])
    callbacks = [instantiate(cfg) for cfg in callback_configs]

    # Initialize loggers
    loggers = []
    wandb_run = None

    # WandB initialization (only if flag is True)
    if USE_WANDB:
        try:
            log.info("Initializing WandB...")
            import wandb
            
            # Initialize wandb directly with hardcoded values
            wandb_run = wandb.init(
                project="emg2qwerty",
                entity="alvister88",
                name=f"emg2qwerty-{wandb.util.generate_id()}",  # Generate a unique name
                config=OmegaConf.to_container(config, resolve=True),  # Log all config
                tags=["pytorch-lightning", "emg2qwerty"]
            )
            log.info(f"WandB initialized with run name: {wandb_run.name}")
            
            # Create a PyTorch Lightning WandB logger that uses the existing run
            wandb_logger = pl.loggers.WandbLogger(
                experiment=wandb_run,
                log_model=True,  # Enable model checkpointing
            )
            loggers.append(wandb_logger)
            
            # Watch model parameters and gradients
            if hasattr(module, "model"):
                wandb_logger.watch(module.model, log="all", log_freq=100)
                log.info("Model parameters being tracked in WandB")
                
        except Exception as e:
            log.error(f"Failed to initialize WandB: {e}", exc_info=True)
            log.info("Will continue without WandB logging")
    
    # Always add a CSV logger for local logging
    csv_logger = pl.loggers.CSVLogger(save_dir=os.path.join(os.getcwd(), "logs"))
    loggers.append(csv_logger)
    log.info("CSV logger initialized for local logging")

    # Initialize trainer with loggers
    trainer = pl.Trainer(
        **config.trainer,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=20,
        enable_checkpointing=True,
    )

    if config.train:
        # Check if a past checkpoint exists to resume training from
        checkpoint_dir = Path.cwd().joinpath("checkpoints")
        resume_from_checkpoint = utils.get_last_checkpoint(checkpoint_dir)
        if resume_from_checkpoint is not None:
            log.info(f"Resuming training from checkpoint {resume_from_checkpoint}")

        # Train
        trainer.fit(module, datamodule, ckpt_path=resume_from_checkpoint)

        # Load best checkpoint
        best_checkpoint_path = trainer.checkpoint_callback.best_model_path
        if os.path.exists(best_checkpoint_path):
            log.info(f"Loading best checkpoint: {best_checkpoint_path}")
            module = module.load_from_checkpoint(
                best_checkpoint_path,
                optimizer=config.optimizer,
                lr_scheduler=config.lr_scheduler,
                decoder=config.decoder,
            )
        else:
            log.warning(f"Best checkpoint not found: {best_checkpoint_path}")

    # Validate and test on the best checkpoint (if training), or on the
    # loaded `config.checkpoint` (otherwise)
    val_metrics = trainer.validate(module, datamodule)
    test_metrics = trainer.test(module, datamodule)

    results = {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_checkpoint": trainer.checkpoint_callback.best_model_path if hasattr(trainer, "checkpoint_callback") else None,
    }
    pprint.pprint(results, sort_dicts=False)
    
    # Log final results to WandB if enabled
    if USE_WANDB and wandb_run:
        try:
            import wandb
            # Log best checkpoint as artifact
            best_checkpoint_path = trainer.checkpoint_callback.best_model_path
            if best_checkpoint_path and os.path.exists(best_checkpoint_path):
                log.info(f"Logging best checkpoint {best_checkpoint_path} to WandB")
                
                # Create and log artifact
                model_artifact = wandb.Artifact(
                    name=f"model-{wandb.run.id}", 
                    type="model",
                    metadata={
                        "val_metrics": val_metrics,
                        "test_metrics": test_metrics
                    }
                )
                model_artifact.add_file(best_checkpoint_path)
                wandb.log_artifact(model_artifact)
                
                # Also log a summary of the final metrics
                wandb.run.summary.update({
                    "final/val_loss": val_metrics[0].get("val/loss", None),
                    "final/val_CER": val_metrics[0].get("val/CER", None),
                    "final/val_WER": val_metrics[0].get("val/WER", None),
                    "final/test_loss": test_metrics[0].get("test/loss", None),
                    "final/test_CER": test_metrics[0].get("test/CER", None),
                    "final/test_WER": test_metrics[0].get("test/WER", None)
                })
            
            # Finish the run
            log.info("Finalizing WandB run")
            wandb.finish()
        except Exception as e:
            log.error(f"Error logging to WandB: {e}", exc_info=True)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("cpus_per_task", utils.cpus_per_task)
    main()