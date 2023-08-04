import os
import csv
import sqlite3
import argparse


'''Insert an LJspeech-Formatted Dataset into the database'''

def create_text_files_and_generate_metadata(dataset_path, db_path, dataset_name, language, name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    sample_path = os.path.join(dataset_path, 'wavs')

    # Create tables if they don't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS dataset (
                      ID INTEGER PRIMARY KEY,
                      name TEXT)''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS speaker (
                      ID INTEGER PRIMARY KEY,
                      name TEXT,
                      dataset_id INTEGER,
                      FOREIGN KEY (dataset_id) REFERENCES dataset(ID))''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS sample (
                      ID INTEGER PRIMARY KEY,
                      txt_path TEXT,
                      audio_path TEXT,
                      speaker_id INTEGER,
                      text TEXT,
                      dataset_id INTEGER,
                      FOREIGN KEY (speaker_id) REFERENCES speaker(ID))''')

    # Insert dataset into the database and get its ID
    cursor.execute("INSERT INTO dataset (name) VALUES (?)", (dataset_name,))
    dataset_id = cursor.lastrowid

    # Insert speaker into the database and get its ID
    speaker_name = name  # Replace this with the actual speaker name
    cursor.execute("INSERT INTO speaker (name, dataset_id) VALUES (?, ?, ?)", (speaker_name, dataset_id, language))
    speaker_id = cursor.lastrowid

    metadata = []
    with open(os.path.join(dataset_path, 'metadata.csv'), newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='|')
        next(reader)  # Skip the header row
        for (filename, transcription) in reader:
            audio_path = os.path.join(dataset_path, 'wavs', filename)
            text_path = os.path.join(sample_path, os.path.splitext(filename)[0] + '.txt')

            # Insert sample information into the database
            cursor.execute("INSERT INTO sample (txt_path, audio_path, speaker_id, text) "
                           "VALUES (?, ?, ?, ?)", (text_path, audio_path, speaker_id, transcription))

            # Get the auto-incrementing sample_id of the inserted row
            sample_id = cursor.lastrowid

            with open(text_path, 'w', encoding='utf-8') as text_file:
                text_file.write(transcription)
            metadata.append({
                'sample_id': sample_id,
                'text_path': text_path,
                'audio_path': audio_path,
                'speaker_id': speaker_id,
                'transcription': transcription,
                'dataset_id': dataset_id
            })

    conn.commit()
    conn.close()

    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument("--ds_name", type=str, default="LJSpeech")
    parser.add_argument("--Language", type=str, default="en")
    parser.add_argument("--speaker_name", type=str, default="LJSpeech")
    args = parser.parse_args()

    metadata = create_text_files_and_generate_metadata(args.dataset_path, args.db_path,args.dsname, args.language, args.speaker_name)
