import customtkinter as ctk
from tkinter import filedialog, messagebox
from pathlib import Path
from PIL import Image
import sqlite3
from app.database import get_base_data_dir, WorkspaceDatabase
from app.engine import MediaFaceEngine

class HansGalleryApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("HansGalleryApp - Portable Media Workspace System")
        self.geometry("1000x650")
        self.engine = MediaFaceEngine()
        self.active_workspace = None
        # Setup Default Root Workspace
        data_dir = get_base_data_dir()
        default_ws = data_dir / "workspaces" / "DefaultWorkspace"
        self.active_workspace = WorkspaceDatabase(default_ws)
        # Grid Architecture Strategy
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.create_sidebar()
        self.create_main_container()
        self.recognition_state = None  # None = face list, 'detail' = showing media for a face
        self.selected_person_id = None
        self.switch_view("gallery")
        
    def create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(5, weight=1)
        self.lbl_title = ctk.CTkLabel(self.sidebar, text="Hans Gallery",
        font=ctk.CTkFont(size=18, weight="bold"))
        self.lbl_title.grid(row=0, column=0, padx=20, pady=25)
        self.btn_gallery_view = ctk.CTkButton(self.sidebar, text="Media Library",
        command=lambda: self.switch_view("gallery"))
        self.btn_gallery_view.grid(row=1, column=0, padx=20, pady=10)
        self.btn_recognition_view = ctk.CTkButton(self.sidebar, text="Face Recognition",
        command=lambda: self.switch_view("recognition"))
        self.btn_recognition_view.grid(row=2, column=0, padx=20, pady=10)
        self.btn_switch_ws = ctk.CTkButton(self.sidebar, text="Switch Catalog",
        fg_color="#4a5568", command=self.action_switch_workspace)
        self.btn_switch_ws.grid(row=3, column=0, padx=20, pady=10)
        self.switch_label = ctk.CTkLabel(self.sidebar, text="Face Tracking Core:")
        self.switch_label.grid(row=4, column=0, padx=20, pady=(20,0))
        self.face_switch = ctk.CTkSwitch(self.sidebar, text="Enabled")
        self.face_switch.select()
        self.face_switch.grid(row=5, column=0, padx=20, pady=5, sticky="n")
        self.lbl_ws_status = ctk.CTkLabel(self.sidebar, text="Workspace: Default",
        font=ctk.CTkFont(size=10))
        self.lbl_ws_status.grid(row=6, column=0, padx=20, pady=15)
        
    def create_main_container(self):
        self.main_content = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        self.main_content.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        self.main_content.grid_columnconfigure(0, weight=1)
        self.main_content.grid_rowconfigure(0, weight=1)
        
    def switch_view(self, target_view):
        for widget in self.main_content.winfo_children():
            widget.destroy()
        if target_view == "gallery":
            self.render_gallery_view()
        elif target_view == "recognition":
            self.render_recognition_view()
            
    def render_gallery_view(self):
        view_frame = ctk.CTkFrame(self.main_content, fg_color="transparent")
        view_frame.grid(row=0, column=0, sticky="nsew")
        view_frame.grid_columnconfigure(0, weight=1)
        view_frame.grid_rowconfigure(1, weight=1)
        top_bar = ctk.CTkFrame(view_frame, height=50)
        top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        btn_import = ctk.CTkButton(top_bar, text="Scan Media Folder",
        command=self.action_import_folder)
        btn_import.pack(side="left", padx=15, pady=10)
        btn_refresh = ctk.CTkButton(top_bar, text="Refresh Catalog",
        command=self.action_refresh_catalog)
        btn_refresh.pack(side="left", padx=5, pady=10)
        # Scrollable Thumbnail Grid
        self.scroll_grid = ctk.CTkScrollableFrame(view_frame)
        self.scroll_grid.grid(row=1, column=0, sticky="nsew")
        self.reload_gallery_thumbnails()
        
    def reload_gallery_thumbnails(self):
        for w in self.scroll_grid.winfo_children():
            w.destroy()
        conn = self.active_workspace.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT thumbnail_name, file_name FROM media")
        items = cursor.fetchall()
        conn.close()
        row, col = 0, 0
        for thumb_name, title in items:
            t_path = self.active_workspace.thumb_dir / thumb_name
            if t_path.exists():
                try:
                    # Load image and maintain aspect ratio
                    img = Image.open(t_path)
                    img.thumbnail((110, 110), Image.Resampling.LANCZOS)
                    img_obj = ctk.CTkImage(light_image=img, size=(img.width, img.height))
                    lbl = ctk.CTkLabel(self.scroll_grid, image=img_obj, text="")
                    lbl.grid(row=row, column=col, padx=12, pady=12)
                except Exception:
                    pass
            col += 1
            if col > 5:
                col = 0
                row += 1
                
    def render_recognition_view(self):
        # Reset state when entering recognition view
        self.recognition_state = None
        self.selected_person_id = None
        self.render_face_list_view()
    
    def render_face_list_view(self):
        """Display list of face clusters (categories)."""
        view_frame = ctk.CTkFrame(self.main_content, fg_color="transparent")
        view_frame.grid(row=0, column=0, sticky="nsew")
        view_frame.grid_columnconfigure(0, weight=1)
        view_frame.grid_rowconfigure(1, weight=1)
        
        # Top bar with title and optimization button
        top_bar = ctk.CTkFrame(view_frame, height=50)
        top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        top_bar.grid_columnconfigure(0, weight=1)
        
        lbl_title = ctk.CTkLabel(top_bar, text="Face Categories", font=ctk.CTkFont(size=14, weight="bold"))
        lbl_title.pack(side="left", padx=15, pady=10)
        
        btn_frame = ctk.CTkFrame(top_bar, fg_color="transparent")
        btn_frame.pack(side="right", padx=15, pady=10)
        
        btn_training = ctk.CTkButton(btn_frame, text="Training Data", fg_color="#2d3748", width=120,
                                     command=self.render_training_data_view)
        btn_training.pack(side="left", padx=5)
        
        btn_optimize = ctk.CTkButton(btn_frame, text="Validate Faces", fg_color="#4a5568", width=120,
                                     command=self.action_show_optimization)
        btn_optimize.pack(side="left", padx=5)
        
        # Scrollable grid of clusters
        scroll_frame = ctk.CTkScrollableFrame(view_frame)
        scroll_frame.grid(row=1, column=0, sticky="nsew")
        scroll_frame.grid_columnconfigure(0, weight=1)
        
        clusters = self.engine.get_all_clusters()
        
        if not clusters:
            lbl_empty = ctk.CTkLabel(scroll_frame, text="No face categories detected yet.\nScan media folders to detect faces.")
            lbl_empty.pack(pady=40)
            return
        
        # Create a frame for the cluster grid
        grid_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        grid_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        row, col = 0, 0
        for cluster_id, cluster_name, cluster_type, count in clusters:
            display_name = cluster_name if cluster_type == 'named' else f"Unknown Person {cluster_id}"
            faces = self.engine.get_cluster_faces(cluster_id, is_named=(cluster_type == 'named'))
            crop_path = faces[0][1] if faces else None
            
            cluster_card = self.create_cluster_card(grid_frame, cluster_id, display_name, crop_path, count, cluster_type)
            cluster_card.grid(row=row, column=col, padx=10, pady=10, sticky="ew")
            grid_frame.grid_columnconfigure(col, weight=1)
            
            col += 1
            if col > 4:
                col = 0
                row += 1
    
    def create_cluster_card(self, parent, cluster_id, display_name, crop_path, count, cluster_type):
        """Create a card widget for a face cluster."""
        card = ctk.CTkFrame(parent, fg_color="#2b2b2b", corner_radius=8)
        
        # Cluster image
        if crop_path and Path(crop_path).exists():
            try:
                img = Image.open(crop_path)
                img.thumbnail((120, 120), Image.Resampling.LANCZOS)
                img_obj = ctk.CTkImage(light_image=img, size=(img.width, img.height))
                lbl_img = ctk.CTkLabel(card, image=img_obj, text="")
                lbl_img.pack(padx=8, pady=8)
                lbl_img.image = img_obj
            except Exception:
                lbl_placeholder = ctk.CTkLabel(card, text="No image", width=120, height=120)
                lbl_placeholder.pack(padx=8, pady=8)
        else:
            lbl_placeholder = ctk.CTkLabel(card, text="No image", width=120, height=120)
            lbl_placeholder.pack(padx=8, pady=8)
        
        # Cluster name and count
        lbl_name = ctk.CTkLabel(card, text=display_name, font=ctk.CTkFont(size=11, weight="bold"), wraplength=120)
        lbl_name.pack(padx=8, pady=4)
        
        lbl_count = ctk.CTkLabel(card, text=f"{count} faces", font=ctk.CTkFont(size=9))
        lbl_count.pack(padx=8, pady=2)
        
        # Buttons frame
        btn_frame = ctk.CTkFrame(card, fg_color="transparent")
        btn_frame.pack(padx=8, pady=8, fill="x")
        
        btn_view = ctk.CTkButton(btn_frame, text="View Media", width=100,
                                command=lambda: self.action_show_cluster_media(cluster_id, display_name, cluster_type))
        btn_view.pack(fill="x", pady=4)
        
        return card
    
    def action_show_cluster_media(self, cluster_id, cluster_name, cluster_type):
        """Show all media files containing faces from the cluster."""
        self.selected_cluster_id = cluster_id
        self.selected_cluster_name = cluster_name
        self.selected_cluster_type = cluster_type
        self.render_cluster_detail_view()
    
    def render_cluster_detail_view(self):
        """Display all media files in a cluster."""
        view_frame = ctk.CTkFrame(self.main_content, fg_color="transparent")
        view_frame.grid(row=0, column=0, sticky="nsew")
        view_frame.grid_columnconfigure(0, weight=1)
        view_frame.grid_rowconfigure(1, weight=1)
        
        # Top bar with back button
        top_bar = ctk.CTkFrame(view_frame, height=50)
        top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        btn_back = ctk.CTkButton(top_bar, text="← Back to Categories", width=150,
                                command=self.action_back_to_clusters)
        btn_back.pack(side="left", padx=15, pady=10)
        
        lbl_title = ctk.CTkLabel(top_bar, text=f"Media: {self.selected_cluster_name}", font=ctk.CTkFont(size=14, weight="bold"))
        lbl_title.pack(side="left", padx=15, pady=10)
        
        # Scrollable grid of media
        scroll_frame = ctk.CTkScrollableFrame(view_frame)
        scroll_frame.grid(row=1, column=0, sticky="nsew")
        
        # Get all encoding IDs for this cluster
        if self.selected_cluster_type == 'named':
            cluster_faces = self.engine.get_cluster_faces(self.selected_cluster_id, is_named=True)
        else:
            cluster_faces = self.engine.get_cluster_faces(self.selected_cluster_id, is_named=False)
        
        # Get media files for all these encoding IDs
        media_files = self.engine.get_media_files_for_encoding_ids([face[0] for face in cluster_faces], self.active_workspace)
        
        if not media_files:
            lbl_empty = ctk.CTkLabel(scroll_frame, text="No media files found for this category.")
            lbl_empty.pack(pady=40)
            return
        
        # Create grid of thumbnails
        row, col = 0, 0
        for media_id, file_name, thumb_name in media_files:
            t_path = self.active_workspace.thumb_dir / thumb_name
            if t_path.exists():
                try:
                    img = Image.open(t_path)
                    img.thumbnail((110, 110), Image.Resampling.LANCZOS)
                    img_obj = ctk.CTkImage(light_image=img, size=(img.width, img.height))
                    lbl = ctk.CTkLabel(scroll_frame, image=img_obj, text="")
                    lbl.grid(row=row, column=col, padx=12, pady=12)
                except Exception:
                    pass
            col += 1
            if col > 5:
                col = 0
                row += 1
    
    def action_back_to_clusters(self):
        """Return to the cluster list view."""
        self.render_face_list_view()
    
    def render_training_data_view(self):
        """Display all named people (training data categories)."""
        view_frame = ctk.CTkFrame(self.main_content, fg_color="transparent")
        view_frame.grid(row=0, column=0, sticky="nsew")
        view_frame.grid_columnconfigure(0, weight=1)
        view_frame.grid_rowconfigure(1, weight=1)
        
        # Top bar with back button
        top_bar = ctk.CTkFrame(view_frame, height=50)
        top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        top_bar.grid_columnconfigure(1, weight=1)
        
        btn_back = ctk.CTkButton(top_bar, text="← Back", width=100,
                                command=lambda: self.render_face_list_view())
        btn_back.grid(row=0, column=0, padx=15, pady=10)
        
        lbl_info = ctk.CTkLabel(top_bar, text="Training Data - Confirmed Categories", font=ctk.CTkFont(size=14, weight="bold"))
        lbl_info.grid(row=0, column=1, padx=15, pady=10, sticky="w")
        
        btn_recluster = ctk.CTkButton(top_bar, text="Recluster Faces", fg_color="#2f855a", width=130,
                                     command=self.action_recluster_faces)
        btn_recluster.grid(row=0, column=2, padx=15, pady=10)
        
        # Get all named people
        people = self.engine.get_unique_people()
        
        # Scrollable content
        content_frame = ctk.CTkScrollableFrame(view_frame)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10, columnspan=3)
        content_frame.grid_columnconfigure(0, weight=1)
        
        if not people:
            lbl_empty = ctk.CTkLabel(content_frame, text="No training data yet.\nValidate faces to build training categories.")
            lbl_empty.pack(pady=40)
            return
        
        # Show each person with their face count
        for person_id, person_name in people:
            # Get count of training faces for this person
            conn = sqlite3.connect(str(self.engine.global_db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM global_encodings WHERE person_id = ?", (person_id,))
            count = cursor.fetchone()[0]
            conn.close()
            
            # Create a button for each person
            btn = ctk.CTkButton(content_frame, 
                               text=f"{person_name} ({count} training faces)",
                               height=50,
                               command=lambda pid=person_id, pname=person_name: self.render_person_training_view(pid, pname))
            btn.pack(fill="x", pady=5)
    
    def render_person_training_view(self, person_id, person_name):
        """Display all training faces for a specific person."""
        view_frame = ctk.CTkFrame(self.main_content, fg_color="transparent")
        view_frame.grid(row=0, column=0, sticky="nsew")
        view_frame.grid_columnconfigure(0, weight=1)
        view_frame.grid_rowconfigure(1, weight=1)
        
        # Top bar with back button
        top_bar = ctk.CTkFrame(view_frame, height=50)
        top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        btn_back = ctk.CTkButton(top_bar, text="← Back", width=100,
                                command=self.render_training_data_view)
        btn_back.pack(side="left", padx=15, pady=10)
        
        lbl_info = ctk.CTkLabel(top_bar, text=f"Training Faces - {person_name}", font=ctk.CTkFont(size=14, weight="bold"))
        lbl_info.pack(side="left", padx=15, pady=10)
        
        # Get all training faces for this person
        conn = sqlite3.connect(str(self.engine.global_db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT id, crop_path FROM global_encodings WHERE person_id = ? ORDER BY id", (person_id,))
        faces = cursor.fetchall()
        conn.close()
        
        # Scrollable content
        content_frame = ctk.CTkScrollableFrame(view_frame)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        content_frame.grid_columnconfigure(0, weight=1)
        
        if not faces:
            lbl_empty = ctk.CTkLabel(content_frame, text="No training faces for this person.")
            lbl_empty.pack(pady=40)
            return
        
        # Display faces in a grid
        for idx, (face_id, crop_path) in enumerate(faces):
            face_container = ctk.CTkFrame(content_frame, fg_color="transparent")
            face_container.pack(fill="x", pady=5, padx=10)
            face_container.grid_columnconfigure(0, weight=1)
            
            # Image
            img_frame = ctk.CTkFrame(face_container)
            img_frame.grid(row=0, column=0, sticky="w", padx=10)
            
            if Path(crop_path).exists():
                try:
                    img = Image.open(crop_path)
                    img.thumbnail((80, 80), Image.Resampling.LANCZOS)
                    img_obj = ctk.CTkImage(light_image=img, size=(img.width, img.height))
                    lbl_img = ctk.CTkLabel(img_frame, image=img_obj, text="")
                    lbl_img.pack()
                    lbl_img.image = img_obj
                except Exception:
                    lbl_img = ctk.CTkLabel(img_frame, text="No image", width=80, height=80)
                    lbl_img.pack()
            
            # Info and delete button
            info_frame = ctk.CTkFrame(face_container, fg_color="transparent")
            info_frame.grid(row=0, column=1, sticky="ew", padx=20)
            info_frame.grid_columnconfigure(0, weight=1)
            
            lbl_info = ctk.CTkLabel(info_frame, text=f"Face #{face_id}", font=ctk.CTkFont(size=11))
            lbl_info.pack(side="left")
            
            btn_delete = ctk.CTkButton(info_frame, text="Delete from Training", fg_color="#c53030", width=150,
                                      command=lambda fid=face_id, pid=person_id, pname=person_name: self.action_delete_training_face(fid, pid, pname))
            btn_delete.pack(side="right", padx=10)
    
    def action_delete_training_face(self, face_id, person_id, person_name):
        """Delete a face from training data."""
        if messagebox.askyesno("Confirm Delete", "Remove this face from training data?\nThe model will be retrained."):
            # Get face path before deletion
            conn = sqlite3.connect(str(self.engine.global_db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT crop_path FROM global_encodings WHERE id = ?", (face_id,))
            result = cursor.fetchone()
            crop_path = result[0] if result else None
            
            # Delete face file
            if crop_path and Path(crop_path).exists():
                try:
                    Path(crop_path).unlink()
                except Exception:
                    pass
            
            # Delete from database and create new predicted cluster
            cursor.execute("""
            UPDATE global_encodings 
            SET person_id = NULL, predicted_cluster_id = ?
            WHERE id = ?
            """, (face_id, face_id))  # Use face_id as cluster ID for unconfirmed faces
            
            conn.commit()
            conn.close()
            
            # Retrain model
            self.engine.train_global_classifier()
            
            messagebox.showinfo("Deleted", "Face removed from training data.\nModel retrained.")
            self.render_person_training_view(person_id, person_name)
    
    def action_recluster_faces(self):
        """Recluster all unconfirmed faces using the trained model."""
        # Show progress dialog
        progress_dialog = ctk.CTkToplevel(self)
        progress_dialog.title("Reclustering Faces")
        progress_dialog.geometry("400x150")
        progress_dialog.update()
        progress_dialog.after(100, progress_dialog.grab_set)
        
        lbl_status = ctk.CTkLabel(progress_dialog, text="Reclustering faces using training data...", font=ctk.CTkFont(size=12))
        lbl_status.pack(padx=20, pady=10)
        
        progress_bar = ctk.CTkProgressBar(progress_dialog)
        progress_bar.set(0.3)
        progress_bar.pack(padx=20, pady=10, fill="x")
        
        progress_dialog.update()
        
        try:
            # Animate progress bar
            progress_bar.set(0.6)
            progress_dialog.update()
            
            # Recluster using trained model
            self.engine.cluster_faces()
            
            progress_bar.set(1.0)
            progress_dialog.update()
            progress_dialog.after(500, progress_dialog.destroy)
            
            messagebox.showinfo("Success", "Faces reclustered using training data!\nCategories have been updated.")
            self.render_face_list_view()
        except Exception as e:
            progress_dialog.destroy()
            messagebox.showerror("Error", f"Failed to recluster faces: {str(e)}")
    
    def action_show_optimization(self):
        """Show the face validation screen."""
        self.render_validation_view()
    
    def render_validation_view(self):
        """Display a single unconfirmed face for the user to assign to a cluster."""
        view_frame = ctk.CTkFrame(self.main_content, fg_color="transparent")
        view_frame.grid(row=0, column=0, sticky="nsew")
        view_frame.grid_columnconfigure(0, weight=1)
        view_frame.grid_rowconfigure(1, weight=1)
        
        # Top bar with back button
        top_bar = ctk.CTkFrame(view_frame, height=50)
        top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        btn_back = ctk.CTkButton(top_bar, text="← Back", width=100,
                                command=lambda: self.render_face_list_view())
        btn_back.pack(side="left", padx=15, pady=10)
        
        lbl_info = ctk.CTkLabel(top_bar, text="Validate Face", font=ctk.CTkFont(size=14, weight="bold"))
        lbl_info.pack(side="left", padx=15, pady=10)
        
        # Get next unconfirmed face
        face_data = self.engine.get_unconfirmed_face()
        if not face_data:
            lbl_empty = ctk.CTkLabel(view_frame, text="All faces have been validated!\nGreat job training the model.")
            lbl_empty.pack(pady=40)
            return
        
        face_id, crop_path, predicted_cluster_id = face_data
        
        # Main content
        content_frame = ctk.CTkScrollableFrame(view_frame)
        content_frame.grid(row=1, column=0, sticky="nsew")
        content_frame.grid_columnconfigure(0, weight=1)
        
        # Display face
        face_container = ctk.CTkFrame(content_frame, fg_color="transparent")
        face_container.pack(pady=20)
        
        lbl_prompt = ctk.CTkLabel(face_container, text="Which category does this face belong to?",
                                 font=ctk.CTkFont(size=12, weight="bold"))
        lbl_prompt.pack(pady=10)
        
        if Path(crop_path).exists():
            try:
                img = Image.open(crop_path)
                img.thumbnail((150, 150), Image.Resampling.LANCZOS)
                img_obj = ctk.CTkImage(light_image=img, size=(img.width, img.height))
                lbl_img = ctk.CTkLabel(face_container, image=img_obj, text="")
                lbl_img.pack(pady=10)
                lbl_img.image = img_obj
            except Exception:
                pass
        
        # Show cluster options
        clusters = self.engine.get_all_clusters()
        
        btn_frame = ctk.CTkScrollableFrame(content_frame)
        btn_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        for cluster_id, cluster_name, cluster_type, count in clusters:
            display_name = cluster_name if cluster_type == 'named' else f"Unknown Person {cluster_id}"
            btn = ctk.CTkButton(btn_frame, text=f"{display_name} ({count} faces)",
                               command=lambda cid=cluster_id, ct=cluster_type: self.action_assign_face_to_cluster(face_id, cid, ct))
            btn.pack(fill="x", pady=5)
        
        # Option to create new cluster
        lbl_or = ctk.CTkLabel(btn_frame, text="Or create new category:", font=ctk.CTkFont(weight="bold"))
        lbl_or.pack(pady=10)
        
        entry_new = ctk.CTkEntry(btn_frame, placeholder_text="New category name")
        entry_new.pack(fill="x", pady=5)
        
        btn_create = ctk.CTkButton(btn_frame, text="Create New Category", fg_color="#2f855a",
                                  command=lambda: self.action_create_cluster_for_face(face_id, entry_new.get()))
        btn_create.pack(fill="x", pady=5)
        
        # Skip option
        btn_skip = ctk.CTkButton(btn_frame, text="Skip This Face", fg_color="#c53030",
                                command=lambda: self.render_validation_view())
        btn_skip.pack(fill="x", pady=5)
    
    def action_assign_face_to_cluster(self, face_id, cluster_id, cluster_type):
        """Assign a face to an existing cluster."""
        if cluster_type == 'named':
            # Assigning to a confirmed person
            if self.engine.assign_face_to_cluster(face_id, cluster_id):
                self.engine.update_workspace_faces_for_encoding(face_id, cluster_id, self.active_workspace)
                messagebox.showinfo("Assigned", "Face assigned to cluster!")
                self.render_validation_view()
            else:
                messagebox.showerror("Error", "Failed to assign face.")
        else:
            # Assigning to an unnamed cluster - need to ask for name
            dialog = ctk.CTkToplevel(self)
            dialog.title("Name This Person")
            dialog.geometry("300x150")
            dialog.update()
            dialog.after(100, dialog.grab_set)
            
            lbl_text = ctk.CTkLabel(dialog, text="Give this person a name:")
            lbl_text.pack(padx=10, pady=10)
            
            entry_name = ctk.CTkEntry(dialog, placeholder_text="Person name")
            entry_name.pack(padx=10, pady=5, fill="x")
            
            def confirm_and_assign():
                name = entry_name.get().strip()
                if not name:
                    messagebox.showwarning("Error", "Please enter a name.")
                    return
                # Create a new person with this name
                person_id = self.engine.create_person_from_faces(name, [face_id], self.active_workspace)
                if person_id:
                    messagebox.showinfo("Created", f"Created '{name}' and assigned face!")
                    dialog.destroy()
                    self.render_validation_view()
                else:
                    messagebox.showerror("Error", "Failed to create person.")
            
            btn_confirm = ctk.CTkButton(dialog, text="Confirm", command=confirm_and_assign)
            btn_confirm.pack(padx=10, pady=10, fill="x")
    
    def action_create_cluster_for_face(self, face_id, name):
        """Create a new named cluster and assign the face to it."""
        if not name.strip():
            messagebox.showwarning("Error", "Please enter a category name.")
            return
        person_id = self.engine.create_person_from_faces(name.strip(), [face_id], self.active_workspace)
        if person_id:
            messagebox.showinfo("Created", f"Created '{name}' and assigned face!")
            self.render_validation_view()
        else:
            messagebox.showerror("Error", "Failed to create category.")
    
    def action_update_face_name(self, person_id, entry_widget):
        """Save the updated name for a face."""
        new_name = entry_widget.get().strip()
        if not new_name:
            messagebox.showwarning("Update Name", "Name cannot be empty.")
            return
        if self.engine.update_person_name(person_id, new_name):
            messagebox.showinfo("Update Name", f"Face name updated to '{new_name}'.")
            self.render_face_list_view()
        else:
            messagebox.showerror("Update Name", "Failed to update name.")
    
    def action_import_folder(self):
        target_dir = filedialog.askdirectory()
        if not target_dir:
            return
        p = Path(target_dir)
        valid_exts = ('.jpg', '.jpeg', '.png')
        files = [f for f in p.rglob('*') if f.is_file() and f.suffix.lower() in valid_exts]
        if not files:
            messagebox.showinfo("Scanner Engine", "No valid image files matching formats located inside specified directory path.")
            return
        
        # Create loading dialog with progress bar
        progress_dialog = ctk.CTkToplevel(self)
        progress_dialog.title("Scanning Media Folder")
        progress_dialog.geometry("400x150")
        progress_dialog.grab_set()
        
        lbl_status = ctk.CTkLabel(progress_dialog, text="Starting scan...", font=ctk.CTkFont(size=12))
        lbl_status.pack(padx=20, pady=10)
        
        progress_bar = ctk.CTkProgressBar(progress_dialog)
        progress_bar.set(0)
        progress_bar.pack(padx=20, pady=10, fill="x")
        
        lbl_count = ctk.CTkLabel(progress_dialog, text="0 / 0", font=ctk.CTkFont(size=10))
        lbl_count.pack(padx=20, pady=5)
        
        use_face = self.face_switch.get()
        success_count = 0
        total_files = len(files)
        
        for idx, f in enumerate(files):
            lbl_status.configure(text=f"Processing: {f.name}")
            lbl_count.configure(text=f"{idx + 1} / {total_files}")
            progress_bar.set((idx + 1) / total_files)
            progress_dialog.update()
            
            if self.engine.process_media_file(f, self.active_workspace, use_face):
                success_count += 1
        
        # Cluster faces after scanning
        if use_face:
            lbl_status.configure(text="Clustering faces...")
            progress_bar.set(0.9)
            progress_dialog.update()
            self.engine.cluster_faces()
        
        progress_dialog.destroy()
        messagebox.showinfo("Scanner Engine", f"Scan operation complete.\nSuccessfully cataloged {success_count} assets.")
        self.reload_gallery_thumbnails()
    
    def action_refresh_catalog(self):
        """Refresh the catalog by removing entries for deleted files."""
        deleted_count = self.engine.cleanup_deleted_files(self.active_workspace)
        messagebox.showinfo("Catalog Refreshed", f"Removed {deleted_count} entries for deleted files.\nCatalog cleaned up.")
        self.reload_gallery_thumbnails()
        
    def action_switch_workspace(self):
        target_ws_dir = filedialog.askdirectory(title="Select Catalog Folder Workspace Storage Path")
        if not target_ws_dir:
            return
        p = Path(target_ws_dir)
        self.active_workspace = WorkspaceDatabase(p)
        self.lbl_ws_status.configure(text=f"Workspace: {p.name}")
        self.switch_view("gallery")
