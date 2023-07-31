# main.py (or any other desired filename)

from fastapi import FastAPI
from typing import List
from voice_smith.utils.pydantic_models import TrainingRun
from voice_smith.config.globals import DB_PATH

import sqlite3

app = FastAPI()

# Function to fetch rows from the "training_run" table
def fetch_training_runs():
    db_path = DB_PATH  # Change this to your SQLite database file path
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM training_run")
        rows = cursor.fetchall()
    # Convert each row (tuple) to a dictionary
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in rows]

# FastAPI GET endpoint to fetch all rows from the "training_run" table
@app.get("/training_run/", response_model=List[TrainingRun])
def get_training_runs():
    training_runs = fetch_training_runs()
    return [TrainingRun(**row) for row in training_runs]

if __name__ == "__main__":
    import uvicorn

    # Run the server using uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
