import json
import os
from pathlib import Path

import torch
from accelerate import Accelerator


TRAINING_STATE_DIRNAME = "training_state"
TRAINER_STATE_FILENAME = "trainer_state.json"


def load_trainer_state(input_dir):
    trainer_state_path = Path(input_dir) / TRAINER_STATE_FILENAME
    if not trainer_state_path.exists():
        raise FileNotFoundError(f"Trainer state file not found: {trainer_state_path}")
    return json.loads(trainer_state_path.read_text())


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0

    def _training_state_path(self, checkpoint_name):
        return os.path.join(self.output_path, TRAINING_STATE_DIRNAME, checkpoint_name)

    def save_training_state(self, accelerator: Accelerator, checkpoint_name, trainer_state=None):
        state_dir = self._training_state_path(checkpoint_name)
        accelerator.wait_for_everyone()
        accelerator.save_state(state_dir)
        if accelerator.is_main_process and trainer_state is not None:
            path = Path(state_dir)
            path.mkdir(parents=True, exist_ok=True)
            trainer_state_path = path / TRAINER_STATE_FILENAME
            trainer_state_path.write_text(json.dumps(trainer_state, indent=2, sort_keys=True) + "\n")

    def on_step_end(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        save_steps=None,
        global_step=None,
        trainer_state=None,
        **kwargs,
    ):
        del kwargs
        if global_step is None:
            self.num_steps += 1
        else:
            self.num_steps = int(global_step)
        if save_steps is not None and self.num_steps % save_steps == 0:
            checkpoint_name = f"step-{self.num_steps}"
            self.save_model(
                accelerator,
                model,
                f"{checkpoint_name}.safetensors",
                training_state_name=checkpoint_name,
                trainer_state=trainer_state,
            )

    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id, trainer_state=None):
        checkpoint_name = f"epoch-{epoch_id}"
        self.save_model(
            accelerator,
            model,
            f"{checkpoint_name}.safetensors",
            training_state_name=checkpoint_name,
            trainer_state=trainer_state,
        )

    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, trainer_state=None):
        if save_steps is not None and self.num_steps % save_steps > 0:
            checkpoint_name = f"step-{self.num_steps}"
            self.save_model(
                accelerator,
                model,
                f"{checkpoint_name}.safetensors",
                training_state_name=checkpoint_name,
                trainer_state=trainer_state,
            )

    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name, training_state_name=None, trainer_state=None):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
        if training_state_name is not None:
            self.save_training_state(accelerator, training_state_name, trainer_state=trainer_state)
