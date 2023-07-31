from pydantic import BaseModel

# Pydantic model for the "training_run" table
class TrainingRun(BaseModel):
    ID: int
    stage: str = "not_started"
    maximum_workers: int
    name: str
    validation_size: float
    min_seconds: float
    max_seconds: float
    use_audio_normalization: bool
    acoustic_learning_rate: float
    acoustic_training_iterations: int
    acoustic_batch_size: int
    acoustic_grad_accum_steps: int
    acoustic_validate_every: int
    device: str = "CPU"
    vocoder_learning_rate: float
    vocoder_training_iterations: int
    vocoder_batch_size: int
    vocoder_grad_accum_steps: int
    vocoder_validate_every: int
    preprocessing_stage: str = "not_started"
    preprocessing_copying_files_progress: float = 0.0
    preprocessing_gen_vocab_progress: float = 0.0
    preprocessing_gen_align_progress: float = 0.0
    preprocessing_extract_data_progress: float = 0.0
    acoustic_fine_tuning_progress: float = 0.0
    ground_truth_alignment_progress: float = 0.0
    vocoder_fine_tuning_progress: float = 0.0
    save_model_progress: float = 0.0
    only_train_speaker_emb_until: int
    skip_on_error: bool = True
    forced_alignment_batch_size: int = 200000
    acoustic_model_type: str = "multilingual"
    dataset_id: int = None

# Pydantic model for the "sample_to_align" table
class SampleToAlign(BaseModel):
    ID: int
    sample_id: int
    training_run_id: int = None
    sample_splitting_run_id: int = None

# Pydantic model for the "dataset" table
class Dataset(BaseModel):
    ID: int
    name: str

# Pydantic model for the "speaker" table
class Speaker(BaseModel):
    ID: int
    name: str
    language: str = "en"
    dataset_id: int

# Pydantic model for the "sample" table
class Sample(BaseModel):
    ID: int
    txt_path: str
    audio_path: str
    speaker_id: int
    text: str

# Pydantic model for the "image_statistic" table
class ImageStatistic(BaseModel):
    ID: int
    name: str
    step: int
    stage: str
    training_run_id: int

# Pydantic model for the "audio_statistic" table
class AudioStatistic(BaseModel):
    ID: int
    name: str
    step: int
    stage: str
    training_run_id: int

# Pydantic model for the "graph_statistic" table
class GraphStatistic(BaseModel):
    ID: int
    name: str
    step: int
    stage: str
    value: float
    training_run_id: int

# Pydantic model for the "model" table
class Model(BaseModel):
    ID: int
    name: str
    type: str
    description: str
    created_at: str

# Pydantic model for the "model_speaker" table
class ModelSpeaker(BaseModel):
    ID: int
    name: str
    speaker_id: int
    model_id: int

# Pydantic model for the "lexicon_word" table
class LexiconWord(BaseModel):
    ID: int
    word: str
    phonemes: str
    model_id: int

# Pydantic model for the "symbol" table
class Symbol(BaseModel):
    ID: int
    symbol: str
    symbol_id: int
    model_id: int

# Pydantic model for the "audio_synth" table
class AudioSynth(BaseModel):
    ID: int
    file_name: str
    text: str
    speaker_name: str
    model_name: str
    created_at: str
    sampling_rate: int
    dur_secs: float

# Pydantic model for the "cleaning_run" table
class CleaningRun(BaseModel):
    ID: int
    name: str
    copying_files_progress: float = 0.0
    transcription_progress: float = 0.0
    applying_changes_progress: float = 0.0
    skip_on_error: bool = True
    stage: str = "not_started"
    device: str = "CPU"
    maximum_workers: int
    dataset_id: int = None

# Pydantic model for the "cleaning_run_sample" table
class CleaningRunSample(BaseModel):
    ID: int
    quality_score: float = None
    sample_id: int
    transcription: str
    cleaning_run_id: int

# Pydantic model for the "text_normalization_run" table
class TextNormalizationRun(BaseModel):
    ID: int
    name: str
    stage: str = "not_started"
    text_normalization_progress: float = 0.0
    dataset_id: int = None

# Pydantic model for the "text_normalization_sample" table
class TextNormalizationSample(BaseModel):
    ID: int
    old_text: str
    new_text: str
    reason: str
    sample_id: int
    text_normalization_run_id: int

# Pydantic model for the "settings" table
class Settings(BaseModel):
    ID: int
    data_path: str = None
    pid: int = None

# Pydantic model for the "sample_splitting_run" table
class SampleSplittingRun(BaseModel):
    ID: int
    maximum_workers: int
    name: str
    stage: str = "not_started"
    copying_files_progress: float = 0.0
    gen_vocab_progress: float = 0.0
    gen_align_progress: float = 0.0
    creating_splits_progress: float = 0.0
    applying_changes_progress: float = 0.0
    device: str = "CPU"
    skip_on_error: bool = True
    forced_alignment_batch_size: int = 200000
    dataset_id: int = None

# Pydantic model for the "sample_splitting_run_sample" table
class SampleSplittingRunSample(BaseModel):
    ID: int
    text: str
    sample_splitting_run_id: int
    sample_id: int

# Pydantic model for the "sample_splitting_run_split" table
class SampleSplittingRunSplit(BaseModel):
    ID: int
    text: str
    split_idx: int
    sample_splitting_run_sample_id: int
