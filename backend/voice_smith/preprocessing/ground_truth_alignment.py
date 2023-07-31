import torch
from pathlib import Path
import json
from torch.utils.data import DataLoader
from typing import Dict, Literal, Optional, Union
from voice_smith.utils.model import get_acoustic_models
from voice_smith.utils.tools import to_device, iter_logger
from voice_smith.config.configs import (
    AcousticFinetuningConfig,
    PreprocessingConfig,
    AcousticMultilingualModelConfig as AcousticModelConfig
)
from voice_smith.utils.loggers import Logger
from voice_smith.model.acoustic_model import AcousticModel
from voice_smith.utils.dataset import AcousticDataset


def save_gta(
    db_id: int,
    table_name: str,
    gen: AcousticModel,
    loader: DataLoader,
    id2speaker: Dict[int, str],
    device: torch.device,
    data_dir: str,
    step: Union[Literal["train"], Literal["val"]],
    logger: Optional[Logger],
    log_every: int = 50,
):
    gen.eval()
    output_mels_gta_dir = Path(data_dir) / "data_gta"

    def callback(index: int):
        if logger is None:
            return
        if index % log_every == 0:
            if step == "train":
                progress = (index / len(loader)) * (4 / 5)
            else:
                progress = (4 / 5) + (index / len(loader)) / 5
            logger.query(
                f"UPDATE {table_name} SET ground_truth_alignment_progress=? WHERE id=?",
                [progress, db_id],
            )

    for batchs in iter_logger(loader, cb=callback, total=len(loader)):
        for batch in batchs:
            batch = to_device(batch, device, is_eval=False)
            (
                ids,
                raw_texts,
                speakers,
                speaker_names,
                texts,
                src_lens,
                mels,
                pitches,
                mel_lens,
                langs,
                attn_priors,
            ) = batch
            print(src_lens)
            print(mel_lens)
            with torch.no_grad():
                outputs = gen.forward_train(
                    x=texts,
                    speakers=speakers,
                    src_lens=src_lens,
                    mels=mels,
                    mel_lens=mel_lens,
                    pitches=pitches,
                    attn_priors=attn_priors,
                    langs=langs,
                )
                y_pred = outputs["y_pred"]
            for basename, speaker_id,speakername, mel_pred, mel_len, mel in zip(
                ids, speakers,speaker_names, y_pred, mel_lens, mels
            ):
                print(speaker_id, speakername)
                speaker_name = speakername#id2speaker[int(speaker_id.item())]
                (output_mels_gta_dir / speaker_name).mkdir(exist_ok=True, parents=True)
                mel_pred = mel_pred[:, :mel_len]
                torch.save(
                    {"mel": mel_pred.cpu()},
                    output_mels_gta_dir / speaker_name / f"{basename}.pt",
                )


def ground_truth_alignment(
    db_id: int,
    table_name: str,
    training_run_name: str,
    batch_size: int,
    group_size: int,
    checkpoint_acoustic: str,
    device: torch.device,
    logger: Optional[Logger],
    assets_path: str,
    training_runs_path: str,
    log_every: int = 200,
):
    print("Generating ground truth aligned data ... \n")
    # TODO change group size automatically
    data_path = Path(training_runs_path) / str(training_run_name) / "data"
    group_size = 5
    dataset = AcousticDataset(
        filename="train.txt",
        batch_size=batch_size,
        sort=False,
        drop_last=True,
        data_path=data_path,
        assets_path=assets_path,
        is_eval=False,
    )
    train_loader = DataLoader(
        dataset,
        num_workers=4,
        batch_size=batch_size * group_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    dataset = AcousticDataset(
        filename="val.txt",
        batch_size=batch_size * group_size,
        sort=False,
        drop_last=False,
        data_path=data_path,
        assets_path=assets_path,
        is_eval=False,
    )
    eval_loader = DataLoader(
        dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    with open(data_path / "speakers.json", "r", encoding="utf-8") as f:
        speakers = json.load(f)

    id2speaker = {speakers[key]: key for key in speakers.keys()}

    gen, optim, step = get_acoustic_models(
        data_path=str(data_path),
        checkpoint_acoustic=checkpoint_acoustic,
        train_config=AcousticFinetuningConfig(),
        preprocess_config=PreprocessingConfig(language="multilingual"),
        model_config=AcousticModelConfig(),
        fine_tuning=True,
        device=device,
        reset=False,
        assets_path=assets_path,
    )

    print("Generating GTA for training set ... \n")
    save_gta(
        db_id=db_id,
        table_name=table_name,
        gen=gen,
        loader=train_loader,
        id2speaker=id2speaker,
        device=device,
        data_dir=str(data_path),
        logger=logger,
        log_every=log_every,
        step="train",
    )

    print("Generating GTA for validation set ... \n")
    save_gta(
        db_id=db_id,
        table_name=table_name,
        gen=gen,
        loader=eval_loader,
        id2speaker=id2speaker,
        device=device,
        data_dir=str(data_path),
        logger=logger,
        log_every=log_every,
        step="val",
    )

    logger.query(
        f"UPDATE {table_name} SET ground_truth_alignment_progress=? WHERE id=?",
        [1.0, db_id],
    )
