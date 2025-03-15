# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

import wandb
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

class EMG2QWERTYWandbLogger(pl.loggers.WandbLogger):
    """
    Weights & Biases logger for EMG2QWERTY project.
    
    This logger extends PyTorch Lightning's WandbLogger with project-specific
    configuration and artifact tracking.
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        project: str = "emg2qwerty",
        entity: Optional[str] = None,
        save_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        log_model: bool = True,
        mode: str = "online",
        **kwargs
    ):
        """
        Initialize the WandB logger for EMG2QWERTY.
        
        Args:
            name: Run name. If None, uses a randomly generated name.
            project: WandB project name.
            entity: WandB team/username.
            save_dir: Directory to save WandB files.
            config: Additional config parameters to log.
            tags: List of tags for this run.
            notes: Optional notes for this run.
            log_model: If True, log model checkpoints as artifacts.
            mode: WandB mode (online, offline, disabled).
            **kwargs: Additional arguments to pass to `wandb.init()`.
        """
        log.info(f"Initializing EMG2QWERTYWandbLogger with: project={project}, entity={entity}, mode={mode}")
        
        if config and isinstance(config, DictConfig):
            # Convert OmegaConf DictConfig to standard dict for WandB
            config = OmegaConf.to_container(config, resolve=True)
        
        # Initialize the parent WandbLogger
        try:
            super().__init__(
                name=name,
                project=project,
                entity=entity,
                save_dir=save_dir,
                config=config,
                tags=tags,
                notes=notes,
                log_model=log_model,
                mode=mode,
                **kwargs
            )
            log.info(f"WandbLogger successfully initialized: {self.name}")
        except Exception as e:
            log.error(f"Error initializing WandbLogger: {e}", exc_info=True)
            raise
        
        self.log_hyperparams = self._log_hyperparams

    def _log_hyperparams(self, params):
        """
        Override hyperparameter logging to handle OmegaConf objects.
        """
        try:
            if isinstance(params, DictConfig):
                params = OmegaConf.to_container(params, resolve=True)
            return super().log_hyperparams(params)
        except Exception as e:
            log.error(f"Error logging hyperparameters: {e}", exc_info=True)
    
    def watch_model(self, model, log_freq=100):
        """
        Watch model gradients and parameters.
        
        Args:
            model: PyTorch model to watch.
            log_freq: Frequency of logging.
        """
        try:
            self.experiment.watch(model, log="all", log_freq=log_freq)
            log.info(f"Model watching enabled with log_freq={log_freq}")
        except Exception as e:
            log.error(f"Error watching model: {e}", exc_info=True)
    
    def log_config(self, config: DictConfig):
        """
        Log the full configuration to WandB.
        
        Args:
            config: The Hydra configuration object.
        """
        try:
            # Convert to a regular dict that wandb can handle
            config_dict = OmegaConf.to_container(config, resolve=True)
            self.experiment.config.update(config_dict)
            log.info("Configuration successfully logged to WandB")
        except Exception as e:
            log.error(f"Error logging config: {e}", exc_info=True)
    
    def log_metrics_dict(self, metrics_dict, step=None):
        """
        Log multiple metrics at once.
        
        Args:
            metrics_dict: Dictionary of metrics to log.
            step: Optional step number.
        """
        try:
            self.experiment.log(metrics_dict, step=step)
        except Exception as e:
            log.error(f"Error logging metrics: {e}", exc_info=True)
    
    def log_checkpoint(self, checkpoint_path: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Log a checkpoint as a WandB artifact.
        
        Args:
            checkpoint_path: Path to the checkpoint file.
            metadata: Optional metadata to attach to the artifact.
        """
        try:
            if not os.path.exists(checkpoint_path):
                log.error(f"Checkpoint file not found: {checkpoint_path}")
                return
                
            artifact = wandb.Artifact(
                name=f"checkpoint-{self.experiment.id}",
                type="model",
                metadata=metadata
            )
            artifact.add_file(checkpoint_path)
            self.experiment.log_artifact(artifact)
            log.info(f"Checkpoint logged as artifact: {checkpoint_path}")
        except Exception as e:
            log.error(f"Error logging checkpoint: {e}", exc_info=True)
    
    def log_sample_predictions(self, predictions, targets, step=None):
        """
        Log sample predictions and targets for visualization.
        
        Args:
            predictions: List of predicted character sequences.
            targets: List of target character sequences.
            step: Optional step number.
        """
        try:
            # Create a table of predictions vs targets
            table = wandb.Table(columns=["Ground Truth", "Prediction"])
            for target, pred in zip(targets, predictions):
                table.add_data(target, pred)
            
            self.experiment.log({"prediction_samples": table}, step=step)
            log.info(f"Logged {len(predictions)} sample predictions to WandB")
        except Exception as e:
            log.error(f"Error logging sample predictions: {e}", exc_info=True)
    
    def log_error_rates(self, cer, wer, step=None):
        """
        Log character and word error rates.
        
        Args:
            cer: Character error rate.
            wer: Word error rate.
            step: Optional step number.
        """
        try:
            self.experiment.log({
                "character_error_rate": cer,
                "word_error_rate": wer
            }, step=step)
            log.info(f"Logged error rates to WandB: CER={cer}, WER={wer}")
        except Exception as e:
            log.error(f"Error logging error rates: {e}", exc_info=True)
    
    def finalize(self, status: str = "success"):
        """
        Finalize the logging process.
        
        Args:
            status: The status of the run. Can be "success", "failed", etc.
        """
        try:
            self.experiment.finish(exit_code=0 if status == "success" else 1)
            log.info(f"WandB run finalized with status: {status}")
        except Exception as e:
            log.error(f"Error finalizing WandB run: {e}", exc_info=True)


# Factory function to easily create the logger from Hydra config
def create_wandb_logger(config: DictConfig) -> EMG2QWERTYWandbLogger:
    """
    Create a WandB logger from Hydra config.
    
    Args:
        config: Hydra configuration.
        
    Returns:
        EMG2QWERTYWandbLogger: Configured WandB logger.
    """
    log.info("Creating WandB logger from config...")
    
    # Extract wandb-specific config if available
    wandb_config = config.get("wandb", {})
    log.info(f"Raw wandb config: {wandb_config}")
    
    # Convert to dict if it's a DictConfig
    if isinstance(wandb_config, DictConfig):
        wandb_config = OmegaConf.to_container(wandb_config, resolve=True)
    
    # Set default parameters if not specified
    wandb_config.setdefault("project", "emg2qwerty")
    wandb_config.setdefault("name", None)  # Will generate a random name
    wandb_config.setdefault("tags", ["pytorch-lightning"])
    
    log.info(f"Processed wandb config: {wandb_config}")
    
    # Create a safe copy of the full config for logging
    full_config = OmegaConf.to_container(config, resolve=True)
    
    # Check if wandb is installed and available
    try:
        import wandb
        log.info(f"WandB version: {wandb.__version__}")
        
        # Test WandB connectivity with a dry run
        test_run = wandb.init(project="test", mode="dryrun")
        test_run.finish()
        log.info("WandB connectivity test successful")
    except ImportError:
        log.error("WandB package not installed. Install with: pip install wandb")
        raise
    except Exception as e:
        log.error(f"WandB connectivity test failed: {e}", exc_info=True)
        raise
    
    # Initialize the logger
    try:
        logger = EMG2QWERTYWandbLogger(
            config=full_config,
            **wandb_config
        )
        
        log.info(f"WandB logger created successfully: {logger.name}")
        return logger
    except Exception as e:
        log.error(f"Failed to create WandB logger: {e}", exc_info=True)
        raise