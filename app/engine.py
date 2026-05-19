import os
import uuid
import pickle
import sqlite3
import numpy as np
from pathlib import Path
from PIL import Image
import face_recognition
from sklearn import svm
from app.database import get_base_data_dir, init_global_db

class MediaFaceEngine:
    def __init__(self):
        self.data_dir = get_base_data_dir()
        self.global_db_path = init_global_db()
        self.model_path = self.data_dir / "global_face_model.pkl"
        self.clf = self.load_model()
        
    def load_model(self):
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None
    
    def train_global_classifier(self):
        """Extracts all registered binary vectors and fits the unified SVM multi-class 
        architecture."""
        conn = sqlite3.connect(str(self.global_db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT person_id, face_encoding FROM global_encodings WHERE person_id IS NOT NULL")
        rows = cursor.fetchall()
        conn.close()
        if len(rows) < 2:
            return False # Minimum mathematical threshold required for SVM partitioning
        encodings = []
        labels = []
        for person_id, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float64)
            if vec.shape[0] == 128:
                encodings.append(vec)
                labels.append(person_id)
        # Check unique constraints
        if len(set(labels)) < 2:
            return False
        try:
            self.clf = svm.SVC(C=1.0, kernel='linear', probability=True, gamma='scale')
            self.clf.fit(encodings, labels)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.clf, f)
            return True
        except Exception:
            return False
    
    def process_media_file(self, original_path: Path, workspace_db, enable_face):
        """Processes an image file by generating thumbnails and calculating facial
        embeddings."""
        try:
            # Generate a fast localized thumbnail
            img = Image.open(original_path)
            img.thumbnail((150, 150))
            thumb_name = f"thumb_{uuid.uuid4().hex}.png"
            img.save(workspace_db.thumb_dir / thumb_name, "PNG")
            w_conn = workspace_db.get_connection()
            w_cursor = w_conn.cursor()
            w_cursor.execute("""
            INSERT OR IGNORE INTO media (file_path, file_name, thumbnail_name, face_scanned)
            VALUES (?, ?, ?, ?)
            """, (str(original_path), original_path.name, thumb_name, 1 if enable_face else 0))
            media_id = w_cursor.lastrowid
            if not media_id:
                w_cursor.execute("SELECT id FROM media WHERE file_path = ?",
                (str(original_path),))
                dia_id = w_cursor.fetchone()[0]
            if enable_face:
                raw_img = face_recognition.load_image_file(str(original_path))
                boxes = face_recognition.face_locations(raw_img, model="hog")
                vecs = face_recognition.face_encodings(raw_img, boxes)
                g_conn = sqlite3.connect(str(self.global_db_path))
                g_cursor = g_conn.cursor()
                for box, vec in zip(boxes, vecs):
                    blob_data = vec.tobytes()
                    crop_uuid = f"crop_{uuid.uuid4().hex}.jpg"
                    crop_path = self.data_dir / "face_crops" / crop_uuid
                    # Extract coordinates and crop face image
                    top, right, bottom, left = box
                    pil_raw = Image.open(original_path)
                    face_crop = pil_raw.crop((left, top, right, bottom))
                    face_crop.resize((120, 120)).save(crop_path, "JPEG")
                    # Store vectors in the system-wide global database as unassigned (-1) initially
                    g_cursor.execute("""
                    INSERT INTO global_encodings (person_id, face_encoding, crop_path)
                    VALUES (NULL, ?, ?)
                    """, (blob_data, str(crop_path)))
                    g_enc_id = g_cursor.lastrowid
                    # Infer dynamic identity vectors via the SVM model
                    predicted_id = None
                    if self.clf is not None:
                        try:
                            pred = self.clf.predict([vec])
                            predicted_id = int(pred[0])
                        except Exception:
                            predicted_id = None
                    w_cursor.execute("""
                    INSERT INTO workspace_faces (media_id, box_top, box_right, box_bottom,
                    box_left, global_person_id, global_encoding_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (media_id, top, right, bottom, left, predicted_id, g_enc_id))
                g_conn.commit()
                g_conn.close()
            w_conn.commit()
            w_conn.close()
            return True
        except Exception:
            return False
        
    def fetch_active_learning_pair(self):
        """Pulls two unlinked or unknown face crops from the global database for human-in-theloop verification."""
        conn = sqlite3.connect(str(self.global_db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT id, crop_path, face_encoding FROM global_encodings ORDER BY RANDOM() LIMIT 2")
        rows = cursor.fetchall()
        conn.close()
        if len(rows) < 2:
            return None
        return rows
    
    def merge_identities_into_profile(self, enc_id_1, enc_id_2, name_string):
        """Binds selected target records to a validated identity framework and forces an SVM
        recalculation."""
        g_conn = sqlite3.connect(str(self.global_db_path))
        g_cursor = g_conn.cursor()
        g_cursor.execute("INSERT OR IGNORE INTO people (name) VALUES (?)", (name_string,))
        g_cursor.execute("SELECT id FROM people WHERE name = ?", (name_string,))
        person_id = g_cursor.fetchone()[0]
        # Route vectors inside both parameters to target identity profile ID
        g_cursor.execute("UPDATE global_encodings SET person_id = ? WHERE id IN (?, ?)", (person_id, enc_id_1, enc_id_2))
        g_conn.commit()
        g_conn.close()
        # Re-trigger pipeline model training calculations asynchronously or contextually
        self.train_global_classifier()
        return person_id
    
    def get_unique_people(self):
        """Returns list of all unique people with their IDs and names."""
        conn = sqlite3.connect(str(self.global_db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM people ORDER BY name")
        rows = cursor.fetchall()
        conn.close()
        return rows
    
    def get_face_crop_for_person(self, person_id):
        """Returns a single face crop path for a person (for preview thumbnail)."""
        conn = sqlite3.connect(str(self.global_db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT crop_path FROM global_encodings WHERE person_id = ? LIMIT 1", (person_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return row[0]
        return None
    
    def get_media_files_with_person(self, person_id, workspace_db):
        """Returns list of media files that contain the specified person."""
        # First, get all encoding IDs for this person from the global database
        g_conn = sqlite3.connect(str(self.global_db_path))
        g_cursor = g_conn.cursor()
        g_cursor.execute("SELECT id FROM global_encodings WHERE person_id = ?", (person_id,))
        encoding_ids = [row[0] for row in g_cursor.fetchall()]
        g_conn.close()
        
        if not encoding_ids:
            return []
        
        return self.get_media_files_for_encoding_ids(encoding_ids, workspace_db)
    
    def get_media_files_for_encoding_ids(self, encoding_ids, workspace_db):
        """Returns list of media files for a list of encoding IDs."""
        if not encoding_ids:
            return []
        
        # Query the workspace database using those encoding IDs
        conn = workspace_db.get_connection()
        cursor = conn.cursor()
        placeholders = ','.join('?' * len(encoding_ids))
        query = f"""
        SELECT DISTINCT m.id, m.file_name, m.thumbnail_name
        FROM media m
        JOIN workspace_faces wf ON m.id = wf.media_id
        WHERE wf.global_encoding_id IN ({placeholders})
        ORDER BY m.file_name
        """
        cursor.execute(query, encoding_ids)
        rows = cursor.fetchall()
        conn.close()
        return rows
    
    def update_person_name(self, person_id, new_name):
        """Updates the name for a person."""
        conn = sqlite3.connect(str(self.global_db_path))
        cursor = conn.cursor()
        try:
            cursor.execute("UPDATE people SET name = ? WHERE id = ?", (new_name, person_id))
            conn.commit()
            conn.close()
            return True
        except Exception:
            conn.close()
            return False
    
    def get_unassigned_faces(self):
        """Returns list of unassigned faces (person_id is NULL)."""
        conn = sqlite3.connect(str(self.global_db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT id, crop_path FROM global_encodings WHERE person_id IS NULL ORDER BY id DESC LIMIT 10")
        rows = cursor.fetchall()
        conn.close()
        return rows
    
    def assign_face_to_person(self, face_encoding_id, person_id):
        """Assigns a face encoding to a person."""
        conn = sqlite3.connect(str(self.global_db_path))
        cursor = conn.cursor()
        try:
            cursor.execute("UPDATE global_encodings SET person_id = ? WHERE id = ?", (person_id, face_encoding_id))
            conn.commit()
            conn.close()
            # Also update workspace_faces table to reflect the assignment
            # This query won't work directly, so we need to do it in the workspace DB
            self.train_global_classifier()
            return True
        except Exception:
            conn.close()
            return False
    
    def update_workspace_faces_for_encoding(self, face_encoding_id, person_id, workspace_db):
        """Update workspace_faces to reflect the person assignment for a given encoding."""
        conn = workspace_db.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
            UPDATE workspace_faces SET global_person_id = ? 
            WHERE global_encoding_id = ?
            """, (person_id, face_encoding_id))
            conn.commit()
            conn.close()
            return True
        except Exception:
            conn.close()
            return False
    
    def create_person_from_faces(self, name, face_encoding_ids, workspace_db=None):
        """Creates a new person and assigns multiple face encodings to them."""
        conn = sqlite3.connect(str(self.global_db_path))
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO people (name) VALUES (?)", (name,))
            person_id = cursor.lastrowid
            for enc_id in face_encoding_ids:
                cursor.execute("UPDATE global_encodings SET person_id = ? WHERE id = ?", (person_id, enc_id))
            conn.commit()
            conn.close()
            # Update workspace_faces if workspace_db is provided
            if workspace_db:
                w_conn = workspace_db.get_connection()
                w_cursor = w_conn.cursor()
                for enc_id in face_encoding_ids:
                    w_cursor.execute("""
                    UPDATE workspace_faces SET global_person_id = ? 
                    WHERE global_encoding_id = ?
                    """, (person_id, enc_id))
                w_conn.commit()
                w_conn.close()
            self.train_global_classifier()
            return person_id
        except Exception:
            conn.close()
            return None
    
    def fetch_pair_with_details(self):
        """Returns two random faces with their details including person_id and names, excluding reviewed pairs."""
        conn = sqlite3.connect(str(self.global_db_path))
        cursor = conn.cursor()
        # Get all faces not in reviewed pairs
        cursor.execute("""
        SELECT g.id, g.crop_path, g.person_id, p.name 
        FROM global_encodings g
        LEFT JOIN people p ON g.person_id = p.id
        WHERE g.id NOT IN (
            SELECT face1_id FROM reviewed_pairs
            UNION
            SELECT face2_id FROM reviewed_pairs
        )
        ORDER BY RANDOM() LIMIT 2
        """)
        rows = cursor.fetchall()
        conn.close()
        if len(rows) < 2:
            return None
        return rows
    
    def mark_pair_as_reviewed(self, face1_id, face2_id, result):
        """Mark a pair as reviewed so we don't show it again."""
        conn = sqlite3.connect(str(self.global_db_path))
        cursor = conn.cursor()
        try:
            # Store with both orders to catch the pair either way
            cursor.execute("""
            INSERT OR IGNORE INTO reviewed_pairs (face1_id, face2_id, result)
            VALUES (?, ?, ?)
            """, (face1_id, face2_id, result))
            conn.commit()
            conn.close()
            return True
        except Exception:
            conn.close()
            return False
    
    def cluster_faces(self):
        """Cluster all unassigned faces using the trained model and assign predicted clusters."""
        conn = sqlite3.connect(str(self.global_db_path))
        cursor = conn.cursor()
        
        # Get all faces that are truly unconfirmed (not validated and not yet predicted)
        cursor.execute("""
        SELECT id, face_encoding FROM global_encodings 
        WHERE predicted_cluster_id IS NULL AND person_id IS NULL
        """)
        rows = cursor.fetchall()
        
        if not rows:
            conn.close()
            return
        
        if self.clf is None:
            # If no trained model, assign each unassigned face its own cluster
            max_cluster = cursor.execute("SELECT MAX(predicted_cluster_id) FROM global_encodings").fetchone()[0] or 0
            for idx, (face_id, _) in enumerate(rows):
                cursor.execute("UPDATE global_encodings SET predicted_cluster_id = ? WHERE id = ?", 
                             (max_cluster + idx + 1, face_id))
            conn.commit()
            conn.close()
            return
        
        # Cluster using the trained model
        max_cluster = cursor.execute("SELECT MAX(predicted_cluster_id) FROM global_encodings").fetchone()[0] or 0
        
        for face_id, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float64)
            if vec.shape[0] != 128:
                continue
            
            try:
                # Try to predict which person this belongs to
                pred = self.clf.predict([vec])
                predicted_person_id = int(pred[0])
                # Use person_id as cluster (confirmed faces)
                cursor.execute("UPDATE global_encodings SET predicted_cluster_id = ? WHERE id = ?", 
                             (predicted_person_id, face_id))
            except Exception:
                # If prediction fails, assign to a new unknown cluster
                max_cluster += 1
                cursor.execute("UPDATE global_encodings SET predicted_cluster_id = ? WHERE id = ?", 
                             (max_cluster, face_id))
        
        conn.commit()
        conn.close()
    
    def get_all_clusters(self):
        """Get all face clusters (both named people and unnamed predicted clusters)."""
        conn = sqlite3.connect(str(self.global_db_path))
        cursor = conn.cursor()
        
        # Get named clusters (from people table)
        cursor.execute("""
        SELECT DISTINCT p.id, p.name, 'named' as type, 
               COUNT(ge.id) as count
        FROM people p
        LEFT JOIN global_encodings ge ON p.id = ge.person_id
        GROUP BY p.id
        """)
        named_clusters = cursor.fetchall()
        
        # Get unnamed predicted clusters
        cursor.execute("""
        SELECT DISTINCT predicted_cluster_id, NULL, 'unnamed' as type,
               COUNT(id) as count
        FROM global_encodings
        WHERE predicted_cluster_id IS NOT NULL 
        AND person_id IS NULL
        GROUP BY predicted_cluster_id
        ORDER BY predicted_cluster_id
        """)
        unnamed_clusters = cursor.fetchall()
        
        conn.close()
        return list(named_clusters) + list(unnamed_clusters)
    
    def get_cluster_faces(self, cluster_id, is_named=True):
        """Get all faces in a cluster. Returns (id, crop_path) tuples."""
        conn = sqlite3.connect(str(self.global_db_path))
        cursor = conn.cursor()
        
        if is_named:
            # Named cluster - faces with person_id set
            cursor.execute("""
            SELECT id, crop_path FROM global_encodings 
            WHERE person_id = ?
            ORDER BY id
            """, (cluster_id,))
        else:
            # Unnamed cluster - faces with predicted_cluster_id set
            cursor.execute("""
            SELECT id, crop_path FROM global_encodings 
            WHERE predicted_cluster_id = ? AND person_id IS NULL
            ORDER BY id
            """, (cluster_id,))
        
        rows = cursor.fetchall()
        conn.close()
        return rows
    
    def get_unconfirmed_face(self):
        """Get a random unconfirmed face for optimization."""
        conn = sqlite3.connect(str(self.global_db_path))
        cursor = conn.cursor()
        # Get faces that haven't been confirmed yet (person_id is NULL)
        cursor.execute("""
        SELECT id, crop_path, predicted_cluster_id
        FROM global_encodings
        WHERE person_id IS NULL
        ORDER BY RANDOM() LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()
        return row
    
    def assign_face_to_cluster(self, face_encoding_id, person_id):
        """Assign a face to a confirmed person cluster."""
        conn = sqlite3.connect(str(self.global_db_path))
        cursor = conn.cursor()
        try:
            cursor.execute("""
            UPDATE global_encodings 
            SET person_id = ?, predicted_cluster_id = NULL
            WHERE id = ?
            """, (person_id, face_encoding_id))
            conn.commit()
            conn.close()
            self.train_global_classifier()
            return True
        except Exception:
            conn.close()
            return False
    
    def cleanup_deleted_files(self, workspace_db):
        """Removes database entries and files for media that no longer exist."""
        conn = workspace_db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, file_path, thumbnail_name FROM media")
        media_items = cursor.fetchall()
        
        deleted_count = 0
        for media_id, file_path, thumb_name in media_items:
            if not Path(file_path).exists():
                # Delete thumbnail file
                thumb_path = workspace_db.thumb_dir / thumb_name
                if thumb_path.exists():
                    try:
                        thumb_path.unlink()
                    except Exception:
                        pass
                
                # Delete associated faces from global encodings
                g_conn = sqlite3.connect(str(self.global_db_path))
                g_cursor = g_conn.cursor()
                g_cursor.execute("""
                SELECT g.id, g.crop_path FROM global_encodings g
                JOIN workspace_faces wf ON g.id = wf.global_encoding_id
                WHERE wf.media_id = ?
                """, (media_id,))
                faces = g_cursor.fetchall()
                
                for face_id, crop_path in faces:
                    if crop_path and Path(crop_path).exists():
                        try:
                            Path(crop_path).unlink()
                        except Exception:
                            pass
                    g_cursor.execute("DELETE FROM global_encodings WHERE id = ?", (face_id,))
                
                g_conn.commit()
                g_conn.close()
                
                # Delete workspace_faces entries
                cursor.execute("DELETE FROM workspace_faces WHERE media_id = ?", (media_id,))
                
                # Delete media entry
                cursor.execute("DELETE FROM media WHERE id = ?", (media_id,))
                deleted_count += 1
        
        conn.commit()
        conn.close()
        self.train_global_classifier()
        return deleted_count
    
    def get_media_count(self, workspace_db):
        """Returns total count of media in workspace."""
        conn = workspace_db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM media")
        count = cursor.fetchone()[0]
        conn.close()
        return count
