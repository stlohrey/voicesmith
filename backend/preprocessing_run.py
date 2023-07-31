from pathlib import Path
import shutil
import json
from typing import Tuple, Callable, Union, Literal
import dataclasses
import argparse
import sqlite3
import torch
from voice_smith.preprocessing.copy_files import copy_files
from voice_smith.preprocessing.extract_data import extract_data
from voice_smith.config.configs import (
    PreprocessLangType,
    PreprocessingConfig,
    AcousticFinetuningConfig,
    AcousticModelConfigType,
    AcousticENModelConfig,
    AcousticMultilingualModelConfig,
    VocoderModelConfig,
    VocoderFinetuningConfig,
)
from voice_smith.utils.sql_logger import SQLLogger
from voice_smith.utils.loggers import set_stream_location
from voice_smith.sql import get_con, save_current_pid
from voice_smith.utils.tools import warnings_to_stdout, get_workers, get_device
from voice_smith.preprocessing.generate_vocab import generate_vocab_mfa
from voice_smith.preprocessing.align import align
from voice_smith.utils.runs import StageRunner
from voice_smith.config.globals import (
    DB_PATH,
    DATASETS_PATH,
    ASSETS_PATH,
    TRAINING_RUNS_PATH,
    MODELS_PATH,
    ENVIRONMENT_NAME,
)


warnings_to_stdout()

# torch.backends.cudnn.benchmark = True


def step_from_ckpt(ckpt: str):
    ckpt_path = Path(ckpt)
    return int(ckpt_path.stem.split("_")[1])


def recalculate_train_size(batch_size: int, grad_acc_step: int, target_size: int):
    if batch_size <= 1:
        raise Exception("Batch size has to be greater than one")

    batch_size = batch_size - 1
    while batch_size * grad_acc_step < target_size:
        grad_acc_step = grad_acc_step + 1
    return batch_size, grad_acc_step


def get_available_name(model_dir: str, name: str):
    model_path = Path(model_dir)
    i = 1
    while (model_path / name).exists():
        name = name + f" ({i})"
        i += 1
    return name


def get_latest_checkpoint(name: str, ckpt_dir: str):
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        return None, 0
    ckpts = [str(el) for el in ckpt_path.iterdir()]
    if len(ckpts) == 0:
        return None, 0
    ckpts = list(map(step_from_ckpt, ckpts))
    ckpts.sort()
    last_ckpt = ckpts[-1]
    last_ckpt_path = ckpt_path / f"{name}_{last_ckpt}.pt"
    return last_ckpt_path, last_ckpt


def model_type_to_language(acoustic_model_type: PreprocessLangType) -> str:
    if acoustic_model_type == "english_only":
        return "english_only"
    elif acoustic_model_type == "multilingual":
        return "multilingual"
    else:
        raise Exception(
            f"No branch selected in switch-statement, {acoustic_model_type} is not a valid case ..."
        )


def model_type_to_acoustic_config(
    acoustic_model_type: PreprocessLangType,
) -> AcousticModelConfigType:
    if acoustic_model_type == "english_only":
        return AcousticENModelConfig()
    elif acoustic_model_type == "multilingual":
        return AcousticMultilingualModelConfig()
    else:
        raise Exception(
            f"No branch selected in switch-statement, {acoustic_model_type} is not a valid case ..."
        )


def get_acoustic_configs(
    cur: sqlite3.Cursor, run_id: int
) -> Tuple[
    PreprocessingConfig,
    Union[AcousticENModelConfig, AcousticMultilingualModelConfig],
    AcousticFinetuningConfig,
]:
    row = cur.execute(
        """
        SELECT min_seconds, max_seconds, maximum_workers, use_audio_normalization, 
        validation_size, acoustic_learning_rate, acoustic_training_iterations, 
        acoustic_batch_size, acoustic_grad_accum_steps, acoustic_validate_every, 
        only_train_speaker_emb_until, forced_alignment_batch_size, skip_on_error, 
        acoustic_model_type
        FROM training_run WHERE ID=?
        """,
        (run_id,),
    ).fetchone()
    (
        min_seconds,
        max_seconds,
        maximum_workers,
        use_audio_normalization,
        validation_size,
        acoustic_learning_rate,
        acoustic_training_iterations,
        batch_size,
        grad_acc_step,
        acoustic_validate_every,
        only_train_speaker_until,
        forced_alignment_batch_size,
        skip_on_error,
        acoustic_model_type,
    ) = row
    p_config: PreprocessingConfig = PreprocessingConfig(
        language=model_type_to_language(acoustic_model_type)
    )
    m_config = model_type_to_acoustic_config(acoustic_model_type)
    t_config: AcousticFinetuningConfig = AcousticFinetuningConfig()
    p_config.val_size = validation_size / 100.0
    p_config.min_seconds = min_seconds
    p_config.max_seconds = max_seconds
    p_config.use_audio_normalization = use_audio_normalization == 1
    p_config.workers = get_workers(maximum_workers)
    p_config.forced_alignment_batch_size = forced_alignment_batch_size
    p_config.skip_on_error = skip_on_error == 1
    t_config.batch_size = batch_size
    t_config.grad_acc_step = grad_acc_step
    t_config.train_steps = acoustic_training_iterations
    t_config.optimizer_config.learning_rate = acoustic_learning_rate
    t_config.val_step = acoustic_validate_every
    t_config.only_train_speaker_until = only_train_speaker_until
    return p_config, m_config, t_config


def get_vocoder_configs(
    cur: sqlite3.Cursor, run_id
) -> Tuple[PreprocessingConfig, VocoderModelConfig, VocoderFinetuningConfig]:
    row = cur.execute(
        """
        SELECT vocoder_learning_rate, vocoder_training_iterations, vocoder_batch_size, vocoder_grad_accum_steps, vocoder_validate_every, acoustic_model_type 
        FROM training_run WHERE ID=?
        """,
        (run_id,),
    ).fetchone()
    (
        vocoder_learning_rate,
        vocoder_training_iterations,
        vocoder_batch_size,
        vocoder_grad_accum_steps,
        vocoder_validate_every,
        acoustic_model_type,
    ) = row
    p_config: PreprocessingConfig = PreprocessingConfig(
        language=model_type_to_language(acoustic_model_type)
    )
    m_config = VocoderModelConfig()
    t_config = VocoderFinetuningConfig()
    t_config.batch_size = vocoder_batch_size
    t_config.grad_accum_steps = vocoder_grad_accum_steps
    t_config.train_steps = vocoder_training_iterations
    t_config.learning_rate = vocoder_learning_rate
    t_config.validation_interval = vocoder_validate_every
    return p_config, m_config, t_config


def not_started_stage(
    cur: sqlite3.Cursor, con: sqlite3.Connection, run_id: int, data_path: str, **kwargs
) -> bool:
    if Path(data_path).exists():
        shutil.rmtree(data_path)
    (Path(data_path) / "logs").mkdir(exist_ok=True, parents=True)
    (Path(data_path) / "raw_data").mkdir(exist_ok=True, parents=True)
    cur.execute(
        "UPDATE training_run SET stage='preprocessing' WHERE ID=?", (run_id,),
    )
    con.commit()
    return False


def preprocessing_stage(
    cur: sqlite3.Cursor,
    con: sqlite3.Connection,
    run_id: int,
    data_path: str,
    dataset_path: str,
    get_logger: Callable[[], SQLLogger],
    assets_path: str,
    environment_name: str,
    training_runs_path: str,
    gen_vocab: bool,
    log_console: bool,
    **kwargs,
) -> bool:
    preprocessing_stage = None
    vocab_path = Path(data_path) / "data" / "vocabs"
    while preprocessing_stage != "finished":
        preprocessing_stage = cur.execute(
            "SELECT preprocessing_stage FROM training_run WHERE ID=?", (run_id,),
        ).fetchone()[0]
        if preprocessing_stage == "not_started":
            set_stream_location(
                str(Path(data_path) / "logs" / "preprocessing.txt"),
                log_console=log_console,
            )
            cur.execute(
                "UPDATE training_run SET preprocessing_stage='copying_files' WHERE ID=?",
                (run_id,),
            )
            con.commit()

        elif preprocessing_stage == "copying_files":
            set_stream_location(
                str(Path(data_path) / "logs" / "preprocessing.txt"),
                log_console=log_console,
            )
            sample_ids, txt_paths, texts, audio_paths, names, langs = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for (
                sample_id,
                txt_path,
                text,
                audio_path,
                speaker_name,
                dataset_id,
                speaker_id,
                lang,
            ) in cur.execute(
                """
                    SELECT sample.ID, sample.txt_path, sample.text, sample.audio_path, 
                    speaker.name AS speaker_name, dataset.ID, speaker.ID, speaker.language 
                    FROM training_run INNER JOIN dataset ON training_run.dataset_id = dataset.ID 
                    INNER JOIN speaker on speaker.dataset_id = dataset.ID
                    INNER JOIN sample on sample.speaker_id = speaker.ID
                    WHERE training_run.ID=?
                    """,
                (run_id,),
            ).fetchall():
                full_audio_path = (
                    Path(dataset_path)
                    / str(dataset_id)
                    / "speakers"
                    / str(speaker_id)
                    / audio_path
                )
                sample_ids.append(sample_id)
                txt_paths.append(txt_path)
                texts.append(text)
                audio_paths.append(str(full_audio_path))
                names.append(speaker_name)
                langs.append(lang)
            p_config, _, _ = get_acoustic_configs(cur=cur, run_id=run_id)

            def progress_cb(progress: float):
                logger = get_logger()
                logger.query(
                    "UPDATE training_run SET preprocessing_copying_files_progress=? WHERE id=?",
                    (progress, run_id),
                )

            copy_files(
                sample_ids=sample_ids,
                data_path=data_path,
                texts=texts,
                langs=langs,
                audio_paths=audio_paths,
                names=names,
                workers=p_config.workers,
                progress_cb=progress_cb,
                skip_on_error=p_config.skip_on_error,
                name_by="name",
            )
            cur.execute(
                "UPDATE training_run SET preprocessing_stage='gen_vocab' WHERE ID=?",
                (run_id,),
            )
            con.commit()

        elif preprocessing_stage == "gen_vocab":
            set_stream_location(
                str(Path(data_path) / "logs" / "preprocessing.txt"),
                log_console=log_console,
            )
            if gen_vocab:
                vocab_path.mkdir(exist_ok=True, parents=True)
                p_config, _, _ = get_acoustic_configs(cur=cur, run_id=run_id)

                row = cur.execute(
                    "SELECT device FROM training_run WHERE ID=?", (run_id,),
                ).fetchone()
                device = row[0]
                device = get_device(device)

                lang_paths = list((Path(data_path) / "raw_data").iterdir())

                for i, lang_path in enumerate(lang_paths):
                    lang = lang_path.name
                    lexica_path = vocab_path / f"{lang}.txt"
                    if lexica_path.exists():
                        continue
                    generate_vocab_mfa(
                        lexicon_path=str(lexica_path),
                        n_workers=p_config.workers,
                        lang=lang,
                        corpus_path=str(lang_path),
                        environment_name=environment_name,
                        language_type=p_config.language,
                    )
                    cur.execute(
                        "UPDATE training_run SET preprocessing_gen_vocab_progress=? WHERE ID=?",
                        ((i + 1) / len(lang_paths), run_id),
                    )
                    con.commit()
            cur.execute(
                "UPDATE training_run SET preprocessing_stage='gen_alignments', preprocessing_gen_vocab_progress=1.0 WHERE ID=?",
                (run_id,),
            )
            con.commit()

        elif preprocessing_stage == "gen_alignments":
            set_stream_location(
                str(Path(data_path) / "logs" / "preprocessing.txt"),
                log_console=log_console,
            )
            p_config, _, _ = get_acoustic_configs(cur=cur, run_id=run_id)
            vocab_paths = list(vocab_path.iterdir())
            for i, vocab_path in enumerate(vocab_paths):
                lang = vocab_path.name.split(".")[0]
                align(
                    cur=cur,
                    con=con,
                    table_name="training_run",
                    foreign_key_name="training_run_id",
                    run_id=run_id,
                    environment_name=environment_name,
                    data_path=data_path,
                    lexicon_path=str(vocab_path),
                    out_path=str(Path(data_path) / "data" / "textgrid" / lang),
                    lang=lang,
                    n_workers=p_config.workers,
                    batch_size=p_config.forced_alignment_batch_size,
                    language_type=p_config.language,
                )
                cur.execute(
                    "UPDATE training_run SET preprocessing_gen_align_progress=? WHERE ID=?",
                    ((i + 1) / len(vocab_paths), run_id),
                )
                con.commit()
            cur.execute(
                "UPDATE training_run SET preprocessing_stage='extract_data', preprocessing_gen_align_progress=1.0 WHERE ID=?",
                (run_id,),
            )
            con.commit()

        elif preprocessing_stage == "extract_data":
            set_stream_location(
                str(Path(data_path) / "logs" / "preprocessing.txt"),
                log_console=log_console,
            )
            row = cur.execute(
                "SELECT validation_size FROM training_run WHERE ID=?", (run_id,),
            ).fetchone()
            p_config, _, _ = get_acoustic_configs(cur=cur, run_id=run_id)
            extract_data(
                db_id=run_id,
                table_name="training_run",
                training_run_name=str(run_id),
                preprocess_config=p_config,
                get_logger=get_logger,
                assets_path=assets_path,
                training_runs_path=training_runs_path,
            )
            cur.execute(
                "UPDATE training_run SET stage='acoustic_fine_tuning', preprocessing_stage='finished' WHERE ID=?",
                (run_id,),
            )
            con.commit()
            """if (Path(data_path) / "raw_data").exists():
                shutil.rmtree(Path(data_path) / "raw_data")"""
    return False


def before_run(data_path: str, **kwargs):
    (Path(data_path) / "logs").mkdir(exist_ok=True, parents=True)

def before_stage(
    data_path: str, stage_name: str, log_console: bool, **kwargs,
):
    set_stream_location(
        str(Path(data_path) / "logs" / f"{stage_name}.txt"), log_console=log_console
    )


def get_stage_name(cur: sqlite3.Cursor, run_id: int, **kwargs):
    row = cur.execute(
        "SELECT stage FROM training_run WHERE ID=?", (run_id,),
    ).fetchone()
    return row[0]


def continue_preprocessing_run(run_id: int, gen_vocab: bool, log_console: bool):
    con = get_con(DB_PATH)
    cur = con.cursor()
    save_current_pid(con=con, cur=cur)
    data_path = Path(TRAINING_RUNS_PATH) / str(run_id)

    def get_logger():
        con = get_con(DB_PATH)
        cur = con.cursor()
        logger = SQLLogger(
            training_run_id=run_id,
            con=con,
            cursor=cur,
            out_dir=str(data_path),
            stage="preprocessing",
        )
        return logger

    runner = StageRunner(
        cur=cur,
        con=con,
        before_run=before_run,
        before_stage=before_stage,
        get_stage_name=get_stage_name,
        stages=[
            ("not_started", not_started_stage),
            ("preprocessing", preprocessing_stage),
        ],
    )
    runner.run(
        cur=cur,
        con=con,
        run_id=run_id,
        data_path=data_path,
        dataset_path=DATASETS_PATH,
        get_logger=get_logger,
        assets_path=ASSETS_PATH,
        environment_name=ENVIRONMENT_NAME,
        training_runs_path=TRAINING_RUNS_PATH,
        models_path=MODELS_PATH,
        log_console=log_console,
        gen_vocab=gen_vocab,
    )
    print(cur,con)
    print(get_stage_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--gen_vocab", action="store_true")
    parser.add_argument("--log_console", action="store_true")
    args = parser.parse_args()
    con = get_con(DB_PATH)
    cur = con.cursor()
    continue_preprocessing_run(run_id=args.run_id,gen_vocab=args.gen_vocab,log_console=args.log_console)
    

