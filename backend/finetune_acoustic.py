import torch
import wandb
from datetime import datetime
from pathlib import Path
from voice_smith.acoustic_training import train_acoustic
from voice_smith.config.configs import (
    PreprocessingConfig,
    AcousticFinetuningConfig,
    AcousticENModelConfig,
)
from voice_smith.utils.wandb_logger import WandBLogger
import argparse
from voice_smith.config.globals import TRAINING_RUNS_PATH, ASSETS_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=f"Finetuning Run at {datetime.now}".replace(":","-"))
    args = parser.parse_args()
    checkpoint_path = None

    if args.checkpoint == None:
        reset = True
        checkpoint_path = str( Path(ASSETS_PATH) / "acoustic_pretrained.pt")
        print(checkpoint_path)
    else:
        checkpoint_path = str(args.checkpoint)
        reset = False

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger = WandBLogger(f"DelightfulTTS Elena finetuning {datetime.now()}".replace(":","-"))
    p_config = PreprocessingConfig(language="multilingual")
    m_config = AcousticENModelConfig()
    t_config = AcousticFinetuningConfig()
    wandb.config.update(
        {
            "preprocess_config": p_config,
            "model_config": m_config,
            "training_config": t_config,
        },
        allow_val_change=True,
    )
    train_acoustic(
        db_id=args.run_id,
        training_run_name=str(args.run_id),
        preprocess_config=p_config,
        model_config=m_config,
        train_config=t_config,
        logger=logger,
        device=device,
        reset=reset,
        checkpoint_acoustic=checkpoint_path,
        fine_tuning=True,
        overwrite_saves=True,
        assets_path=ASSETS_PATH,
        training_runs_path=TRAINING_RUNS_PATH,
    )

