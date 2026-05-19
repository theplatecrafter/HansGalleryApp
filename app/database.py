import os
import sys
import sqlite3
from pathlib import Path

def get_base_data_dir():
    """Resolves the root configuration directory relative to the portable execution source."""
    if getattr(sys, 'frozen', False):
        base_dir = Path(sys.executable).parent
    else:
        base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir

def init_global_db():
    """Initializes the system-wide global facial recognition identity database."""
    data_dir = get_base_data_dir()
    global_db_path = data_dir / "global_face_data.db"
    # Ensure face crops directory exists
    (data_dir / "face_crops").mkdir(exist_ok=True)
    conn = sqlite3.connect(str(global_db_path))
    cursor = conn.cursor()
    # Global explicit profile mapping
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
    )
    """)
    # Global training feature database
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS global_encodings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER,
    predicted_cluster_id INTEGER,
    face_encoding BLOB NOT NULL,
    crop_path TEXT,
    FOREIGN KEY (person_id) REFERENCES people (id) ON DELETE CASCADE
    )
    """)
    # Track reviewed pairs to avoid showing them again
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reviewed_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    face1_id INTEGER NOT NULL,
    face2_id INTEGER NOT NULL,
    result TEXT,
    UNIQUE(face1_id, face2_id)
    )
    """)
    
    # Migrate: Add predicted_cluster_id column if it doesn't exist
    cursor.execute("PRAGMA table_info(global_encodings)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'predicted_cluster_id' not in columns:
        try:
            cursor.execute("ALTER TABLE global_encodings ADD COLUMN predicted_cluster_id INTEGER")
        except sqlite3.OperationalError:
            pass  # Column already exists
    
    conn.commit()
    conn.close()
    return global_db_path

class WorkspaceDatabase:
    """Handles operations and initialization for localized independent workspaces."""
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.db_path = workspace_path / "gallery.db"
        self.thumb_dir = workspace_path / "thumbnails"
        self.thumb_dir.mkdir(parents=True, exist_ok=True)
        self.init_workspace_db()
        
    def init_workspace_db(self):
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        # Local Media file table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS media (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT UNIQUE NOT NULL,
        file_name TEXT NOT NULL,
        thumbnail_name TEXT,
        face_scanned INTEGER DEFAULT 0
        )
        """)
        # Mapping table localized to coordinates but bound to global identities
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS workspace_faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        media_id INTEGER NOT NULL,
        box_top INTEGER NOT NULL,
        box_right INTEGER NOT NULL,
        box_bottom INTEGER NOT NULL,
        box_left INTEGER NOT NULL,
        global_person_id INTEGER,
        global_encoding_id INTEGER,
        FOREIGN KEY (media_id) REFERENCES media (id) ON DELETE CASCADE
        )
        """)
        conn.commit()
        conn.close()
        
    def get_connection(self):
        return sqlite3.connect(str(self.db_path))

