import sqlite3
import argparse
from voice_smith.config.globals import (
    DB_PATH,
)
from voice_smith.sql import get_con

def create_tables(con,cur):
    cur.execute('''
        CREATE TABLE IF NOT EXISTS training_run (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            stage TEXT NOT NULL DEFAULT "not_started",
            maximum_workers INTEGER NOT NULL,
            name TEXT NOT NULL,
            validation_size FLOAT NOT NULL,
            min_seconds FLOAT NOT NULL,
            max_seconds FLOAT NOT NULL, 
            use_audio_normalization BOOLEAN NOT NULL,  
            acoustic_learning_rate FLOAT NOT NULL,
            acoustic_training_iterations BIGINT NOT NULL,
            acoustic_batch_size INTEGER NOT NULL,
            acoustic_grad_accum_steps INTEGER NOT NULL,
            acoustic_validate_every INTEGER NOT NULL,
            device TEXT NOT NULL DEFAULT "CPU", 
            vocoder_learning_rate FLOAT NOT NULL,
            vocoder_training_iterations BIGINT NOT NULL,
            vocoder_batch_size INTEGER NOT NULL,
            vocoder_grad_accum_steps INTEGER NOT NULL,
            vocoder_validate_every INTEGER NOT NULL,
            preprocessing_stage TEXT DEFAULT "not_started" NOT NULL,
            preprocessing_copying_files_progress FLOAT NOT NULL DEFAULT 0.0,
            preprocessing_gen_vocab_progress FLOAT NOT NULL DEFAULT 0.0,
            preprocessing_gen_align_progress FLOAT NOT NULL DEFAULT 0.0,
            preprocessing_extract_data_progress FLOAT NOT NULL DEFAULT 0.0,
            acoustic_fine_tuning_progress FLOAT NOT NULL DEFAULT 0.0,
            ground_truth_alignment_progress FLOAT NOT NULL DEFAULT 0.0,
            vocoder_fine_tuning_progress FLOAT NOT NULL DEFAULT 0.0,
            save_model_progress FLOAT NOT NULL DEFAULT 0.0,
            only_train_speaker_emb_until INTEGER NOT NULL,
            skip_on_error BOOLEAN DEFAULT 1,
            forced_alignment_batch_size INTEGER NOT NULL DEFAULT 200000,
            acoustic_model_type STRING NOT NULL DEFAULT "multilingual",
            dataset_id INTEGER DEFAULT NULL,
            FOREIGN KEY (dataset_id) REFERENCES dataset(ID) ON DELETE SET NULL,
            UNIQUE(name)
        );
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS sample_to_align (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER UNIQUE NOT NULL,
            training_run_id INTEGER DEFAULT NULL,
            sample_splitting_run_id INTEGER DEFAULT NULL,
            FOREIGN KEY (sample_id) REFERENCES sample(ID) ON DELETE CASCADE,
            FOREIGN KEY (training_run_id) REFERENCES training_run(ID) ON DELETE CASCADE,
            FOREIGN KEY (sample_splitting_run_id) REFERENCES sample_splitting_run(ID) ON DELETE CASCADE
        );
    ''')
    cur.execute('''
        CREATE INDEX IF NOT EXISTS sample_to_align_was_aligned_index
        ON sample_to_align(was_aligned)   
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS dataset (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            UNIQUE(name)
        );
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS speaker (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            language TEXT NOT NULL DEFAULT "en",
            dataset_id INTEGER NOT NULL,
            UNIQUE(name, dataset_id),
            FOREIGN KEY (dataset_id) REFERENCES dataset(ID)
        ); 
    ''')
    cur.execute('''
        CREATE INDEX IF NOT EXISTS speaker_language_index
        ON speaker(language)   
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS sample (
            ID INTEGER PRIMARY KEY,
            txt_path TEXT NOT NULL,
            audio_path TEXT NOT NULL,
            speaker_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            UNIQUE(txt_path, speaker_id),
            UNIQUE(audio_path, speaker_id), 
            FOREIGN KEY (speaker_id) REFERENCES speaker(ID)
        );
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS image_statistic (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            step INTEGER NOT NULL,
            stage TEXT NOT NULL,
            training_run_id INTEGER NOT NULL,
            FOREIGN KEY (training_run_id) REFERENCES training_run(ID)
        );
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS audio_statistic (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            step INTEGER NOT NULL,
            stage TEXT NOT NULL,
            training_run_id INTEGER NOT NULL,
            FOREIGN KEY (training_run_id) REFERENCES training_run(ID)
        );
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS graph_statistic (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            step INTEGER NOT NULL,
            stage TEXT NOT NULL,
            value FLOAT NOT NULL,
            training_run_id INTEGER NOT NULL,
            FOREIGN KEY (training_run_id) REFERENCES training_run(ID)
        );
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS model (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            description TEXT NOT NULL,
            created_at DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name)
        );
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS model_speaker (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            speaker_id INTEGER NOT NULL,
            model_id INTEGER NOT NULL,
            FOREIGN KEY (model_id) REFERENCES model(ID) ON DELETE CASCADE,
            UNIQUE(model_id, name),
            UNIQUE(model_id, speaker_id)
        );
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS lexicon_word (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL,
            phonemes TEXT NOT NULL,
            model_id INTEGER NOT NULL,
            FOREIGN KEY (model_id) REFERENCES model(ID) ON DELETE CASCADE,
            UNIQUE(model_id, word)
        );
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS symbol (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            symbol_id INTEGER NOT NULL,
            model_id INTEGER NOT NULL,
            FOREIGN KEY (model_id) REFERENCES model(ID) ON DELETE CASCADE,
            UNIQUE(model_id, symbol),
            UNIQUE(model_id, symbol_id)
        );
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS audio_synth (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            text TEXT NOT NULL,
            speaker_name TEXT NOT NULL,
            model_name TEXT NOT NULL,            
            created_at DEFAULT CURRENT_TIMESTAMP,
            sampling_rate INTEGER NOT NULL,
            dur_secs FLOAT NOT NULL
        );
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS cleaning_run (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            copying_files_progress FLOAT DEFAULT 0.0,
            transcription_progress FLOAT DEFAULT 0.0,
            applying_changes_progress FLOAT DEFAULT 0.0,
            skip_on_error BOOLEAN DEFAULT 1,
            stage TEXT DEFAULT "not_started",
            device TEXT NOT NULL DEFAULT "CPU",
            maximum_workers INTEGER NOT NULL, 
            dataset_id INTEGER DEFAULT NULL,
            FOREIGN KEY (dataset_id) REFERENCES dataset(ID)
        );  
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS cleaning_run_sample (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            quality_score FLOAT DEFAULT NULL,
            sample_id INT NOT NULL,
            transcription TEXT NOT NULL,
            cleaning_run_id INT NOT NULL,
            FOREIGN KEY (sample_id) REFERENCES sample(ID),
            FOREIGN KEY (cleaning_run_id) REFERENCES cleaning_run(ID)
        );
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS text_normalization_run (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            stage TEXT DEFAULT "not_started",
            text_normalization_progress FLOAT DEFAULT 0.0,
            dataset_id INTEGER DEFAULT NULL,  
            FOREIGN KEY (dataset_id) REFERENCES dataset(ID)
        ); 
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS text_normalization_sample (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            old_text TEXT NOT NULL,
            new_text TEXT NOT NULL,
            reason TEXT NOT NULL,
            sample_id INT NOT NULL,
            text_normalization_run_id INT NOT NULL,
            FOREIGN KEY (sample_id) REFERENCES sample(ID),
            FOREIGN KEY (text_normalization_run_id) REFERENCES text_normalization_run(ID)
        );
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            data_path TEXT DEFAULT NULL,
            pid INTEGER DEFAULT NULL
        );
    ''')
    cur.execute('''
        INSERT OR IGNORE INTO settings (ID) VALUES (1) 
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS sample_splitting_run (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            maximum_workers INTEGER NOT NULL,
            name TEXT NOT NULL,
            stage TEXT DEFAULT "not_started",
            copying_files_progress FLOAT NOT NULL DEFAULT 0.0,
            gen_vocab_progress FLOAT NOT NULL DEFAULT 0.0,
            gen_align_progress FLOAT NOT NULL DEFAULT 0.0,
            creating_splits_progress FLOAT NOT NULL DEFAULT 0.0,
            applying_changes_progress FLOAT NOT NULL DEFAULT 0.0,
            device TEXT NOT NULL DEFAULT "CPU",
            skip_on_error BOOLEAN DEFAULT 1,
            forced_alignment_batch_size INTEGER NOT NULL DEFAULT 200000,
            dataset_id INTEGER DEFAULT NULL,
            FOREIGN KEY (dataset_id) REFERENCES dataset(ID)
        );  
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS sample_splitting_run_sample (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            sample_splitting_run_id INTEGER NOT NULL,
            sample_id INTEGER NOT NULL,
            FOREIGN KEY (sample_splitting_run_id) REFERENCES sample_splitting_run(ID),
            FOREIGN KEY (sample_id) REFERENCES sample(ID) 
        ); 
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS sample_splitting_run_split (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            split_idx INTEGER NOT NULL,
            sample_splitting_run_sample_id INTEGER NOT NULL,
            FOREIGN KEY (sample_splitting_run_sample_id) REFERENCES sample_splitting_run_sample(ID) ON DELETE CASCADE
        ); 
    ''')
    con.commit()
    con.close()


if __name__ == "__main__":
     con = get_con(DB_PATH)
     cur = con.cursor()
     create_tables(con,cur)