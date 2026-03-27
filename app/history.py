"""
app/history.py — SQLite database for patient history management.

Manages a local SQLite database storing patient records including:
    - Patient name and age
    - Upload timestamp
    - Predicted skin lesion class and confidence score
    - Path to the uploaded image

Functions:
    init_db()           — Create the patients table if it doesn't exist.
    insert_record()     — Insert a new patient diagnosis record.
    get_all_records()   — Retrieve all records sorted by timestamp.
    get_by_name()       — Search records by patient name (case-insensitive).
"""

import os
import sqlite3
from datetime import datetime
from typing import List, Optional, Tuple

# Database file path (stored in project root)
DB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(DB_DIR, "patient_history.db")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Get a SQLite database connection.

    Args:
        db_path: Path to the SQLite database file.
    Returns:
        sqlite3.Connection object.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable dict-like row access
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """
    Initialize the database by creating the patients table if it doesn't exist.

    Schema:
        id              INTEGER PRIMARY KEY AUTOINCREMENT
        name            TEXT NOT NULL
        age             INTEGER NOT NULL
        timestamp       TEXT NOT NULL (ISO format)
        predicted_class TEXT NOT NULL
        confidence      REAL NOT NULL
        image_path      TEXT NOT NULL
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT NOT NULL,
            age             INTEGER NOT NULL,
            timestamp       TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence      REAL NOT NULL,
            image_path      TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def insert_record(
    name: str,
    age: int,
    predicted_class: str,
    confidence: float,
    image_path: str,
    db_path: str = DB_PATH,
) -> int:
    """
    Insert a new patient diagnosis record into the database.

    Args:
        name: Patient's full name.
        age: Patient's age in years.
        predicted_class: Predicted skin lesion class name.
        confidence: Confidence score (0-100).
        image_path: Path to the saved uploaded image.
        db_path: Path to the database file.
    Returns:
        The auto-generated row ID of the inserted record.
    """
    # Ensure the database is initialized
    init_db(db_path)

    conn = get_connection(db_path)
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute(
        """
        INSERT INTO patients (name, age, timestamp, predicted_class, confidence, image_path)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (name, age, timestamp, predicted_class, confidence, image_path),
    )

    record_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return record_id


def get_all_records(db_path: str = DB_PATH) -> List[dict]:
    """
    Retrieve all patient records from the database, sorted by most recent first.

    Args:
        db_path: Path to the database file.
    Returns:
        List of dictionaries, each representing a patient record.
    """
    init_db(db_path)

    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, name, age, timestamp, predicted_class, confidence, image_path
        FROM patients
        ORDER BY timestamp DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    # Convert sqlite3.Row objects to plain dicts
    records = [dict(row) for row in rows]
    return records


def get_by_name(
    search_name: str,
    db_path: str = DB_PATH,
) -> List[dict]:
    """
    Search for patient records by name (case-insensitive partial match).

    Args:
        search_name: Name or partial name to search for.
        db_path: Path to the database file.
    Returns:
        List of matching patient records as dictionaries.
    """
    init_db(db_path)

    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, name, age, timestamp, predicted_class, confidence, image_path
        FROM patients
        WHERE LOWER(name) LIKE LOWER(?)
        ORDER BY timestamp DESC
        """,
        (f"%{search_name}%",),
    )

    rows = cursor.fetchall()
    conn.close()

    records = [dict(row) for row in rows]
    return records


def delete_record(
    record_id: int,
    db_path: str = DB_PATH,
) -> bool:
    """
    Delete a patient record by ID.

    Args:
        record_id: The ID of the record to delete.
        db_path: Path to the database file.
    Returns:
        True if a record was deleted, False otherwise.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM patients WHERE id = ?", (record_id,))
    deleted = cursor.rowcount > 0

    conn.commit()
    conn.close()

    return deleted


def get_record_count(db_path: str = DB_PATH) -> int:
    """Return the total number of patient records."""
    init_db(db_path)

    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM patients")
    count = cursor.fetchone()[0]
    conn.close()

    return count


if __name__ == "__main__":
    # Quick test
    print("🔧 Initializing database...")
    init_db()

    print("📝 Inserting test records...")
    id1 = insert_record("Alice Smith", 34, "Melanoma", 87.5, "/tmp/img1.jpg")
    id2 = insert_record("Bob Johnson", 52, "Melanocytic nevus", 92.1, "/tmp/img2.jpg")
    id3 = insert_record("Alice Brown", 28, "Basal cell carcinoma", 76.3, "/tmp/img3.jpg")

    print(f"\n📊 Total records: {get_record_count()}")

    print("\n📋 All records:")
    for record in get_all_records():
        print(f"   [{record['id']}] {record['name']}, age {record['age']} — "
              f"{record['predicted_class']} ({record['confidence']:.1f}%)")

    print("\n🔍 Search for 'Alice':")
    for record in get_by_name("Alice"):
        print(f"   [{record['id']}] {record['name']} — {record['predicted_class']}")

    print("\n✅ Database test complete!")
