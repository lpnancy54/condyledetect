#!/usr/bin/env python3
"""
Analyseur de Condyles Mandibulaires
Application GUI pour détecter et mesurer les condyles sur fichiers STL

Auteur: Assistant Claude pour Dr Petitpas
Date: Janvier 2026
"""

import sys
import numpy as np
import json
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QGroupBox, QFormLayout,
    QDoubleSpinBox, QTableWidget, QTableWidgetItem, QSplitter,
    QMessageBox, QStatusBar, QMenuBar, QAction, QToolBar,
    QProgressBar, QTextEdit, QTabWidget, QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QColor

import pyqtgraph.opengl as gl
import pyqtgraph as pg
import trimesh


class CondyleDetector:
    """Classe pour la détection des condyles."""
    
    def __init__(self, search_radius: float = 15.0):
        self.search_radius = search_radius
        self.mesh = None
        self.condyles = {}
        self.measurements = {}
    
    def load_stl(self, filepath: str) -> bool:
        """Charge un fichier STL."""
        try:
            self.mesh = trimesh.load(filepath)
            return True
        except Exception as e:
            print(f"Erreur chargement: {e}")
            return False
    
    def find_condyle_region(self, vertices: np.ndarray, side: str, x_center: float) -> np.ndarray:
        """Trouve la région du condyle pour un côté donné."""
        if side == "gauche":
            side_mask = vertices[:, 0] < x_center
        else:
            side_mask = vertices[:, 0] >= x_center
        
        side_vertices = vertices[side_mask]
        
        if len(side_vertices) < 100:
            return None
        
        # Point de départ supérieur et postérieur pour éviter l'apophyse coronoïde
        z_threshold_seed = np.percentile(side_vertices[:, 2], 90)
        superior_mask = side_vertices[:, 2] >= z_threshold_seed
        superior_points = side_vertices[superior_mask]
        if len(superior_points) == 0:
            return None
        posterior_idx = np.argmax(superior_points[:, 1])
        highest_point = superior_points[posterior_idx]
        
        # Sphère de recherche
        distances = np.linalg.norm(side_vertices - highest_point, axis=1)
        condyle_mask = distances < self.search_radius
        condyle_points = side_vertices[condyle_mask]
        
        # Affiner en favorisant les zones supérieures et surtout postérieures
        # (plus haut et surtout plus en arrière que l'apophyse coronoïde).
        if len(condyle_points) > 50:
            threshold_pairs = [(55, 70), (50, 65), (45, 60)]
            min_points = 20

            refined_points = None

            for z_pct, y_pct in threshold_pairs:
                z_threshold = np.percentile(condyle_points[:, 2], z_pct)
                y_threshold = np.percentile(condyle_points[:, 1], y_pct)

                refined_mask = (condyle_points[:, 2] >= z_threshold) & (condyle_points[:, 1] >= y_threshold)
                candidate = condyle_points[refined_mask]

                if len(candidate) >= min_points:
                    refined_points = candidate
                    break

            if refined_points is not None:
                condyle_points = refined_points
            else:
                z_threshold = np.percentile(condyle_points[:, 2], 45)
                z_only_mask = condyle_points[:, 2] >= z_threshold
                z_only_points = condyle_points[z_only_mask]
                if len(z_only_points) >= min_points:
                    condyle_points = z_only_points
        
        return condyle_points
    
    def compute_geometric_center_3d(self, points: np.ndarray) -> np.ndarray:
        """
        Calcule le centre géométrique 3D du condyle.
        Utilise la bounding box pour trouver le vrai centre volumétrique.
        """
        # Centre de la bounding box (meilleure approximation du centre 3D)
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        bbox_center = (bbox_min + bbox_max) / 2
        
        return bbox_center
    
    def detect(self) -> bool:
        """Détecte les condyles."""
        if self.mesh is None:
            return False
        
        vertices = self.mesh.vertices
        x_center = self.mesh.centroid[0]
        
        self.condyles = {}
        
        for side in ["gauche", "droit"]:
            condyle_points = self.find_condyle_region(vertices, side, x_center)
            
            if condyle_points is None or len(condyle_points) < 20:
                continue
            
            # Centre géométrique 3D (centre de la bounding box)
            geometric_center = self.compute_geometric_center_3d(condyle_points)
            
            # Centroïde des points de surface (pour comparaison)
            surface_centroid = condyle_points.mean(axis=0)
            
            # Bounding box
            bbox_min = condyle_points.min(axis=0)
            bbox_max = condyle_points.max(axis=0)
            
            # Apex
            apex_idx = np.argmax(condyle_points[:, 2])
            apex = condyle_points[apex_idx]
            
            # Pôles médial et latéral
            if side == "gauche":
                medial_idx = np.argmax(condyle_points[:, 0])
                lateral_idx = np.argmin(condyle_points[:, 0])
            else:
                medial_idx = np.argmin(condyle_points[:, 0])
                lateral_idx = np.argmax(condyle_points[:, 0])
            
            medial_pole = condyle_points[medial_idx]
            lateral_pole = condyle_points[lateral_idx]
            
            self.condyles[side] = {
                "centroid": geometric_center,  # Maintenant c'est le vrai centre 3D
                "surface_centroid": surface_centroid,
                "apex": apex,
                "medial_pole": medial_pole,
                "lateral_pole": lateral_pole,
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "width": np.linalg.norm(lateral_pole - medial_pole),
                "num_points": len(condyle_points),
                "points": condyle_points
            }
        
        self._calculate_measurements()
        return len(self.condyles) > 0
    
    def _calculate_measurements(self):
        """Calcule les mesures intercondyliennes."""
        self.measurements = {}
        
        if "gauche" not in self.condyles or "droit" not in self.condyles:
            return
        
        left = self.condyles["gauche"]
        right = self.condyles["droit"]
        
        self.measurements["distance_centres"] = np.linalg.norm(
            right["centroid"] - left["centroid"]
        )
        self.measurements["distance_medial"] = np.linalg.norm(
            right["medial_pole"] - left["medial_pole"]
        )
        self.measurements["distance_lateral"] = np.linalg.norm(
            right["lateral_pole"] - left["lateral_pole"]
        )
        self.measurements["asymetrie_verticale"] = left["centroid"][2] - right["centroid"][2]
        self.measurements["asymetrie_ap"] = left["centroid"][1] - right["centroid"][1]
        
        # Point milieu
        self.measurements["midpoint"] = (left["centroid"] + right["centroid"]) / 2


class MeshViewer(gl.GLViewWidget):
    """Widget de visualisation 3D du maillage."""
    
    def __init__(self):
        super().__init__()
        self.setBackgroundColor('w')
        self.setCameraPosition(distance=150)
        
        # Grille
        grid = gl.GLGridItem()
        grid.scale(10, 10, 1)
        grid.setColor((200, 200, 200, 100))
        self.addItem(grid)
        
        self.mesh_item = None
        self.condyle_items = []
    
    def load_mesh(self, mesh: trimesh.Trimesh):
        """Charge et affiche un maillage."""
        # Supprimer l'ancien maillage
        if self.mesh_item is not None:
            self.removeItem(self.mesh_item)
        
        # Convertir en format pyqtgraph
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Centrer le maillage
        center = mesh.centroid
        vertices = vertices - center
        
        # Créer le mesh item
        mesh_data = gl.MeshData(vertexes=vertices, faces=faces)
        self.mesh_item = gl.GLMeshItem(
            meshdata=mesh_data,
            smooth=True,
            color=(0.8, 0.8, 0.9, 1.0),
            shader='shaded',
            glOptions='opaque'
        )
        self.addItem(self.mesh_item)
        
        # Ajuster la caméra
        bounds = mesh.bounds
        max_dim = max(bounds[1] - bounds[0])
        self.setCameraPosition(distance=max_dim * 1.5)
        
        return center
    
    def show_condyles(self, condyles: dict, center: np.ndarray, measurements: dict):
        """Affiche les marqueurs des condyles en mode filaire avec centres et point milieu."""
        # Supprimer les anciens marqueurs
        for item in self.condyle_items:
            self.removeItem(item)
        self.condyle_items = []
        
        # Couleurs: Vert pour gauche, Rouge pour droit
        colors = {
            "gauche": (0.2, 0.8, 0.2, 0.6),  # Vert semi-transparent
            "droit": (1.0, 0.2, 0.2, 0.6)    # Rouge semi-transparent
        }
        
        center_colors = {
            "gauche": (0.0, 1.0, 0.0, 1.0),  # Vert vif pour le centre
            "droit": (1.0, 0.0, 0.0, 1.0)    # Rouge vif pour le centre
        }
        
        for side, data in condyles.items():
            color = colors[side]
            center_color = center_colors[side]
            
            # Afficher les points du condyle en mode filaire/scatter
            points = data["points"] - center
            
            # Nuage de points semi-transparent (simule le filaire)
            scatter = gl.GLScatterPlotItem(
                pos=points,
                size=3,
                color=color,
                pxMode=True
            )
            self.addItem(scatter)
            self.condyle_items.append(scatter)
            
            # Sphère au CENTRE GÉOMÉTRIQUE 3D (bien visible)
            centroid = data["centroid"] - center
            sphere_center = gl.MeshData.sphere(rows=12, cols=12, radius=3.0)
            item = gl.GLMeshItem(
                meshdata=sphere_center,
                smooth=True,
                color=center_color,
                shader='shaded',
                glOptions='opaque'
            )
            item.translate(*centroid)
            self.addItem(item)
            self.condyle_items.append(item)
        
        # Point MILIEU entre les deux centres condyliens (en jaune)
        if "midpoint" in measurements and measurements["midpoint"] is not None:
            midpoint = measurements["midpoint"] - center
            sphere_mid = gl.MeshData.sphere(rows=12, cols=12, radius=3.5)
            mid_item = gl.GLMeshItem(
                meshdata=sphere_mid,
                smooth=True,
                color=(1.0, 1.0, 0.0, 1.0),  # Jaune vif
                shader='shaded',
                glOptions='opaque'
            )
            mid_item.translate(*midpoint)
            self.addItem(mid_item)
            self.condyle_items.append(mid_item)
            
            # Ligne entre les deux centres (axe bicondylien)
            if "gauche" in condyles and "droit" in condyles:
                left_center = condyles["gauche"]["centroid"] - center
                right_center = condyles["droit"]["centroid"] - center
                
                line_pts = np.array([left_center, right_center])
                line = gl.GLLinePlotItem(
                    pos=line_pts,
                    color=(1.0, 1.0, 0.0, 0.8),
                    width=2.0,
                    antialias=True
                )
                self.addItem(line)
                self.condyle_items.append(line)
    
    def clear_condyles(self):
        """Efface les marqueurs de condyles."""
        for item in self.condyle_items:
            self.removeItem(item)
        self.condyle_items = []


class AnalysisThread(QThread):
    """Thread pour l'analyse en arrière-plan."""
    finished = pyqtSignal(bool)
    progress = pyqtSignal(str)
    
    def __init__(self, detector: CondyleDetector, filepath: str):
        super().__init__()
        self.detector = detector
        self.filepath = filepath
    
    def run(self):
        self.progress.emit("Chargement du fichier STL...")
        if not self.detector.load_stl(self.filepath):
            self.finished.emit(False)
            return
        
        self.progress.emit("Détection des condyles...")
        success = self.detector.detect()
        self.finished.emit(success)


class MainWindow(QMainWindow):
    """Fenêtre principale de l'application."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analyseur de Condyles Mandibulaires")
        self.setGeometry(100, 100, 1400, 900)
        
        self.detector = CondyleDetector()
        self.current_file = None
        self.mesh_center = np.zeros(3)
        
        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()
    
    def _setup_ui(self):
        """Configure l'interface utilisateur."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Splitter principal
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Panneau gauche - Visualisation 3D
        viewer_widget = QWidget()
        viewer_layout = QVBoxLayout(viewer_widget)
        
        viewer_label = QLabel("Visualisation 3D")
        viewer_label.setFont(QFont("Arial", 12, QFont.Bold))
        viewer_layout.addWidget(viewer_label)
        
        self.viewer = MeshViewer()
        viewer_layout.addWidget(self.viewer)
        
        splitter.addWidget(viewer_widget)
        
        # Panneau droit - Contrôles et résultats
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Groupe: Chargement
        load_group = QGroupBox("Fichier")
        load_layout = QVBoxLayout(load_group)
        
        self.file_label = QLabel("Aucun fichier chargé")
        self.file_label.setWordWrap(True)
        load_layout.addWidget(self.file_label)
        
        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("📂 Charger STL")
        self.btn_load.clicked.connect(self.load_file)
        btn_layout.addWidget(self.btn_load)
        
        self.btn_analyze = QPushButton("🔍 Analyser")
        self.btn_analyze.clicked.connect(self.analyze)
        self.btn_analyze.setEnabled(False)
        btn_layout.addWidget(self.btn_analyze)
        
        load_layout.addLayout(btn_layout)
        right_layout.addWidget(load_group)
        
        # Groupe: Paramètres
        params_group = QGroupBox("Paramètres de détection")
        params_layout = QFormLayout(params_group)
        
        self.spin_radius = QDoubleSpinBox()
        self.spin_radius.setRange(5.0, 30.0)
        self.spin_radius.setValue(15.0)
        self.spin_radius.setSuffix(" mm")
        self.spin_radius.setToolTip("Rayon de recherche autour du point le plus haut")
        params_layout.addRow("Rayon de recherche:", self.spin_radius)
        
        right_layout.addWidget(params_group)
        
        # Onglets pour les résultats
        tabs = QTabWidget()
        
        # Onglet: Centres des condyles
        centers_tab = QWidget()
        centers_layout = QVBoxLayout(centers_tab)
        
        self.table_centers = QTableWidget(2, 4)
        self.table_centers.setHorizontalHeaderLabels(["Condyle", "X", "Y", "Z"])
        self.table_centers.setVerticalHeaderLabels(["Gauche", "Droit"])
        self.table_centers.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        centers_layout.addWidget(self.table_centers)
        
        tabs.addTab(centers_tab, "Centres")
        
        # Onglet: Mesures
        measures_tab = QWidget()
        measures_layout = QVBoxLayout(measures_tab)
        
        self.table_measures = QTableWidget(6, 2)
        self.table_measures.setHorizontalHeaderLabels(["Mesure", "Valeur"])
        self.table_measures.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        measures_layout.addWidget(self.table_measures)
        
        tabs.addTab(measures_tab, "Mesures")
        
        # Onglet: Détails
        details_tab = QWidget()
        details_layout = QVBoxLayout(details_tab)
        
        self.text_details = QTextEdit()
        self.text_details.setReadOnly(True)
        self.text_details.setFont(QFont("Consolas", 10))
        details_layout.addWidget(self.text_details)
        
        tabs.addTab(details_tab, "Détails")
        
        right_layout.addWidget(tabs)
        
        # Boutons d'export
        export_group = QGroupBox("Export")
        export_layout = QHBoxLayout(export_group)
        
        self.btn_export_json = QPushButton("💾 JSON")
        self.btn_export_json.clicked.connect(self.export_json)
        self.btn_export_json.setEnabled(False)
        export_layout.addWidget(self.btn_export_json)
        
        self.btn_export_stl = QPushButton("📦 STL marqué")
        self.btn_export_stl.clicked.connect(self.export_stl)
        self.btn_export_stl.setEnabled(False)
        export_layout.addWidget(self.btn_export_stl)
        
        right_layout.addWidget(export_group)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([900, 500])
    
    def _setup_menu(self):
        """Configure la barre de menu."""
        menubar = self.menuBar()
        
        # Menu Fichier
        file_menu = menubar.addMenu("&Fichier")
        
        open_action = QAction("&Ouvrir STL...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.load_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        export_json_action = QAction("Exporter &JSON...", self)
        export_json_action.setShortcut("Ctrl+J")
        export_json_action.triggered.connect(self.export_json)
        file_menu.addAction(export_json_action)
        
        export_stl_action = QAction("Exporter &STL marqué...", self)
        export_stl_action.setShortcut("Ctrl+S")
        export_stl_action.triggered.connect(self.export_stl)
        file_menu.addAction(export_stl_action)
        
        file_menu.addSeparator()
        
        quit_action = QAction("&Quitter", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        
        # Menu Analyse
        analysis_menu = menubar.addMenu("&Analyse")
        
        analyze_action = QAction("&Détecter les condyles", self)
        analyze_action.setShortcut("Ctrl+D")
        analyze_action.triggered.connect(self.analyze)
        analysis_menu.addAction(analyze_action)
        
        # Menu Aide
        help_menu = menubar.addMenu("&Aide")
        
        about_action = QAction("À &propos", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def _setup_toolbar(self):
        """Configure la barre d'outils."""
        toolbar = QToolBar("Outils")
        self.addToolBar(toolbar)
        
        toolbar.addAction("📂 Ouvrir", self.load_file)
        toolbar.addAction("🔍 Analyser", self.analyze)
        toolbar.addSeparator()
        toolbar.addAction("💾 Export JSON", self.export_json)
        toolbar.addAction("📦 Export STL", self.export_stl)
    
    def _setup_statusbar(self):
        """Configure la barre de statut."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Prêt")
        
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(200)
        self.progress.hide()
        self.statusbar.addPermanentWidget(self.progress)
    
    def load_file(self):
        """Ouvre un dialogue pour charger un fichier STL."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Ouvrir un fichier STL",
            "",
            "Fichiers STL (*.stl);;Tous les fichiers (*.*)"
        )
        
        if filepath:
            self.current_file = filepath
            self.file_label.setText(f"Fichier: {Path(filepath).name}")
            self.statusbar.showMessage(f"Chargement de {Path(filepath).name}...")
            
            # Charger et afficher le maillage
            if self.detector.load_stl(filepath):
                self.mesh_center = self.viewer.load_mesh(self.detector.mesh)
                self.btn_analyze.setEnabled(True)
                self.statusbar.showMessage(f"Chargé: {len(self.detector.mesh.vertices)} vertices")
                
                # Effacer les anciens résultats
                self.viewer.clear_condyles()
                self._clear_results()
            else:
                QMessageBox.critical(self, "Erreur", "Impossible de charger le fichier STL")
    
    def analyze(self):
        """Lance l'analyse des condyles."""
        if self.detector.mesh is None:
            return
        
        self.statusbar.showMessage("Analyse en cours...")
        self.progress.setRange(0, 0)
        self.progress.show()
        
        # Mettre à jour le rayon de recherche
        self.detector.search_radius = self.spin_radius.value()
        
        # Détecter les condyles
        if self.detector.detect():
            self._display_results()
            self.viewer.show_condyles(self.detector.condyles, self.mesh_center, self.detector.measurements)
            self.btn_export_json.setEnabled(True)
            self.btn_export_stl.setEnabled(True)
            self.statusbar.showMessage("Analyse terminée - Condyles détectés")
        else:
            QMessageBox.warning(self, "Attention", "Impossible de détecter les condyles")
            self.statusbar.showMessage("Échec de la détection")
        
        self.progress.hide()
    
    def _display_results(self):
        """Affiche les résultats dans les tableaux."""
        condyles = self.detector.condyles
        measurements = self.detector.measurements
        
        # Tableau des centres (ajouter le point milieu)
        self.table_centers.setRowCount(3)
        self.table_centers.setVerticalHeaderLabels(["Gauche", "Droit", "Milieu"])
        
        for row, side in enumerate(["gauche", "droit"]):
            if side in condyles:
                c = condyles[side]["centroid"]
                self.table_centers.setItem(row, 0, QTableWidgetItem(side.capitalize()))
                self.table_centers.setItem(row, 1, QTableWidgetItem(f"{c[0]:.2f}"))
                self.table_centers.setItem(row, 2, QTableWidgetItem(f"{c[1]:.2f}"))
                self.table_centers.setItem(row, 3, QTableWidgetItem(f"{c[2]:.2f}"))
        
        # Point milieu
        if "midpoint" in measurements and measurements["midpoint"] is not None:
            mid = measurements["midpoint"]
            self.table_centers.setItem(2, 0, QTableWidgetItem("Milieu"))
            self.table_centers.setItem(2, 1, QTableWidgetItem(f"{mid[0]:.2f}"))
            self.table_centers.setItem(2, 2, QTableWidgetItem(f"{mid[1]:.2f}"))
            self.table_centers.setItem(2, 3, QTableWidgetItem(f"{mid[2]:.2f}"))
        
        # Tableau des mesures
        measures_data = [
            ("Distance intercondylienne (centres)", f"{measurements.get('distance_centres', 0):.2f} mm"),
            ("Distance pôles médiaux", f"{measurements.get('distance_medial', 0):.2f} mm"),
            ("Distance pôles latéraux", f"{measurements.get('distance_lateral', 0):.2f} mm"),
            ("Asymétrie verticale (G-D)", f"{measurements.get('asymetrie_verticale', 0):.2f} mm"),
            ("Asymétrie antéro-post. (G-D)", f"{measurements.get('asymetrie_ap', 0):.2f} mm"),
            ("Largeur condyle G / D", f"{condyles.get('gauche', {}).get('width', 0):.1f} / {condyles.get('droit', {}).get('width', 0):.1f} mm"),
        ]
        
        self.table_measures.setRowCount(len(measures_data))
        for row, (name, value) in enumerate(measures_data):
            self.table_measures.setItem(row, 0, QTableWidgetItem(name))
            self.table_measures.setItem(row, 1, QTableWidgetItem(value))
        
        # Texte détaillé
        details = "=== RÉSULTATS DE L'ANALYSE ===\n\n"
        
        for side in ["gauche", "droit"]:
            if side in condyles:
                c = condyles[side]
                color_name = "VERT" if side == "gauche" else "ROUGE"
                details += f"CONDYLE {side.upper()} ({color_name})\n"
                details += f"  Centre 3D:   X={c['centroid'][0]:.2f}, Y={c['centroid'][1]:.2f}, Z={c['centroid'][2]:.2f}\n"
                details += f"  Apex:        X={c['apex'][0]:.2f}, Y={c['apex'][1]:.2f}, Z={c['apex'][2]:.2f}\n"
                details += f"  Pôle médial: X={c['medial_pole'][0]:.2f}, Y={c['medial_pole'][1]:.2f}, Z={c['medial_pole'][2]:.2f}\n"
                details += f"  Pôle latéral: X={c['lateral_pole'][0]:.2f}, Y={c['lateral_pole'][1]:.2f}, Z={c['lateral_pole'][2]:.2f}\n"
                details += f"  Largeur:     {c['width']:.2f} mm\n"
                details += f"  Points:      {c['num_points']}\n\n"
        
        if measurements:
            details += "POINT MILIEU INTERCONDYLIEN (JAUNE)\n"
            if "midpoint" in measurements and measurements["midpoint"] is not None:
                mid = measurements["midpoint"]
                details += f"  Position:    X={mid[0]:.2f}, Y={mid[1]:.2f}, Z={mid[2]:.2f}\n\n"
            
            details += "MESURES INTERCONDYLIENNES\n"
            details += f"  Distance centres:    {measurements.get('distance_centres', 0):.2f} mm\n"
            details += f"  Distance médiaux:    {measurements.get('distance_medial', 0):.2f} mm\n"
            details += f"  Distance latéraux:   {measurements.get('distance_lateral', 0):.2f} mm\n"
            details += f"  Asymétrie verticale: {measurements.get('asymetrie_verticale', 0):.2f} mm\n"
            details += f"  Asymétrie A-P:       {measurements.get('asymetrie_ap', 0):.2f} mm\n"
        
        self.text_details.setText(details)
    
    def _clear_results(self):
        """Efface les résultats affichés."""
        self.table_centers.clearContents()
        self.table_measures.clearContents()
        self.text_details.clear()
        self.btn_export_json.setEnabled(False)
        self.btn_export_stl.setEnabled(False)
    
    def export_json(self):
        """Exporte les résultats en JSON."""
        if not self.detector.condyles:
            return
        
        default_name = Path(self.current_file).stem + "_condyles.json" if self.current_file else "condyles.json"
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Exporter en JSON",
            default_name,
            "Fichiers JSON (*.json)"
        )
        
        if filepath:
            export_data = {
                "source_file": self.current_file,
                "parameters": {
                    "search_radius": self.detector.search_radius
                },
                "condyles": {},
                "measurements": {}
            }
            
            for side, data in self.detector.condyles.items():
                export_data["condyles"][side] = {
                    "centroid": data["centroid"].tolist(),
                    "apex": data["apex"].tolist(),
                    "medial_pole": data["medial_pole"].tolist(),
                    "lateral_pole": data["lateral_pole"].tolist(),
                    "width": float(data["width"]),
                    "num_points": data["num_points"]
                }
            
            for key, value in self.detector.measurements.items():
                if isinstance(value, np.ndarray):
                    export_data["measurements"][key] = value.tolist()
                else:
                    export_data["measurements"][key] = float(value)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.statusbar.showMessage(f"Exporté: {filepath}")
    
    def export_stl(self):
        """Exporte le STL avec les marqueurs."""
        if not self.detector.condyles or self.detector.mesh is None:
            return
        
        default_name = Path(self.current_file).stem + "_marked.stl" if self.current_file else "marked.stl"
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Exporter STL marqué",
            default_name,
            "Fichiers STL (*.stl)"
        )
        
        if filepath:
            meshes = [self.detector.mesh.copy()]
            
            for side, data in self.detector.condyles.items():
                # Sphère au centre 3D
                sphere = trimesh.creation.icosphere(subdivisions=2, radius=3.0)
                sphere.apply_translation(data["centroid"])
                meshes.append(sphere)
            
            # Sphère au point milieu
            if "midpoint" in self.detector.measurements:
                mid_sphere = trimesh.creation.icosphere(subdivisions=2, radius=3.5)
                mid_sphere.apply_translation(self.detector.measurements["midpoint"])
                meshes.append(mid_sphere)
            
            combined = trimesh.util.concatenate(meshes)
            combined.export(filepath)
            
            self.statusbar.showMessage(f"Exporté: {filepath}")
    
    def show_about(self):
        """Affiche la boîte de dialogue À propos."""
        QMessageBox.about(
            self,
            "À propos",
            """<h3>Analyseur de Condyles Mandibulaires</h3>
            <p>Version 1.0</p>
            <p>Application de détection et mesure automatique des condyles 
            mandibulaires sur fichiers STL.</p>
            <p><b>Fonctionnalités:</b></p>
            <ul>
                <li>Chargement de fichiers STL</li>
                <li>Détection automatique des condyles</li>
                <li>Calcul des centres géométriques</li>
                <li>Mesures intercondyliennes</li>
                <li>Visualisation 3D interactive</li>
                <li>Export JSON et STL</li>
            </ul>
            <p>Développé pour Dr Petitpas - Janvier 2026</p>"""
        )


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Style sombre optionnel
    # app.setStyleSheet("""
    #     QMainWindow { background-color: #2b2b2b; }
    #     QWidget { color: #ffffff; }
    # """)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
