#!/usr/bin/env python3
"""
Détection précise des condyles mandibulaires dans un fichier STL
et calcul de leurs centres géométriques.

Version améliorée avec meilleure isolation des têtes condyliennes.

Auteur: Assistant Claude pour Dr Petitpas
Date: Janvier 2026
"""

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import json


def load_mandible(filepath: str) -> trimesh.Trimesh:
    """Charge un fichier STL de mandibule."""
    mesh = trimesh.load(filepath)
    print(f"✓ Maillage chargé: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return mesh


def compute_roundness(points: np.ndarray) -> float:
    """
    Calcule la "rondeur" d'un ensemble de points via PCA.
    Retourne un ratio entre 0 et 1 (1 = parfaitement rond).
    """
    if len(points) < 10:
        return 0.0

    # Centrer les points
    centered = points - points.mean(axis=0)

    # PCA pour trouver les axes principaux
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Tri décroissant

    # Ratio du plus petit / plus grand axe
    # Plus proche de 1 = plus rond
    if eigenvalues[0] > 0:
        roundness = eigenvalues[1] / eigenvalues[0]
    else:
        roundness = 0.0

    return roundness


def find_condyle_region(vertices: np.ndarray, side: str, x_center: float) -> np.ndarray:
    """
    Trouve la région du condyle pour un côté donné.

    MÉTHODE: Diviser la branche montante en régions antérieure/postérieure
    et trouver le condyle dans la région POSTÉRIEURE (Y le plus élevé).
    """
    # Séparer par côté
    if side == "gauche":
        side_mask = vertices[:, 0] < x_center
    else:
        side_mask = vertices[:, 0] >= x_center

    side_vertices = vertices[side_mask]

    if len(side_vertices) < 100:
        return None

    # Étape 1: Prendre la région de la branche montante (40% supérieure en Z)
    z_threshold = np.percentile(side_vertices[:, 2], 60)
    ramus_region = side_vertices[side_vertices[:, 2] > z_threshold]

    print(f"  → Branche montante: {len(ramus_region)} points")

    # Étape 2: Analyser la distribution en Y pour trouver les deux sommets
    y_coords = ramus_region[:, 1]
    y_min, y_max = y_coords.min(), y_coords.max()
    y_range = y_max - y_min

    # Trouver le point le plus haut (Z max) pour chaque tiers en Y
    n_sections = 5
    y_sections = np.linspace(y_min, y_max, n_sections + 1)

    section_peaks = []
    for i in range(n_sections):
        mask = (ramus_region[:, 1] >= y_sections[i]) & (ramus_region[:, 1] < y_sections[i + 1])
        section_points = ramus_region[mask]
        if len(section_points) > 10:
            # Trouver le sommet de cette section
            z_max_idx = np.argmax(section_points[:, 2])
            peak_point = section_points[z_max_idx]
            z_max = peak_point[2]
            y_center = (y_sections[i] + y_sections[i + 1]) / 2

            # Calculer la rondeur locale
            top_mask = section_points[:, 2] > np.percentile(section_points[:, 2], 70)
            top_points = section_points[top_mask]
            roundness = compute_roundness(top_points) if len(top_points) > 10 else 0

            section_peaks.append({
                'y_center': y_center,
                'z_max': z_max,
                'peak_point': peak_point,
                'num_points': len(section_points),
                'roundness': roundness,
                'section_idx': i
            })

    print(f"  → {len(section_peaks)} sections analysées")

    if len(section_peaks) == 0:
        # Fallback
        max_z_idx = np.argmax(side_vertices[:, 2])
        highest_point = side_vertices[max_z_idx]
        distances = np.linalg.norm(side_vertices - highest_point, axis=1)
        return side_vertices[distances < 15.0]

    # Étape 3: Trouver les deux pics principaux (coronoïde et condyle)
    # Trier par Z décroissant pour trouver les sommets
    section_peaks.sort(key=lambda s: s['z_max'], reverse=True)

    # Les deux sections les plus hautes sont probablement coronoïde et condyle
    top_sections = section_peaks[:min(3, len(section_peaks))]

    for s in top_sections:
        print(f"    Section Y={s['y_center']:.1f}: Z_max={s['z_max']:.1f}, rondeur={s['roundness']:.2f}")

    # Étape 4: Parmi les pics hauts, le CONDYLE est celui avec Y le plus GRAND (postérieur)
    # ET/OU la meilleure rondeur
    # Score: favoriser Y élevé (postérieur) et bonne rondeur
    for s in top_sections:
        y_score = (s['y_center'] - y_min) / (y_range + 0.001)  # 0 à 1, 1 = postérieur
        s['score'] = 0.7 * y_score + 0.3 * s['roundness']

    # Trier par score
    top_sections.sort(key=lambda s: s['score'], reverse=True)
    condyle_section = top_sections[0]

    print(f"  ✓ Condyle sélectionné: Y={condyle_section['y_center']:.1f} (score={condyle_section['score']:.2f})")

    # Étape 5: Extraire les points autour du pic du condyle
    condyle_y = condyle_section['y_center']
    y_tolerance = y_range / 4  # 25% de la plage

    # Points dans la zone Y du condyle
    y_mask = np.abs(ramus_region[:, 1] - condyle_y) < y_tolerance
    condyle_region = ramus_region[y_mask]

    # Prendre le sommet de cette région
    if len(condyle_region) > 20:
        z_thresh = np.percentile(condyle_region[:, 2], 60)
        condyle_top = condyle_region[condyle_region[:, 2] > z_thresh]
        condyle_center = condyle_top.mean(axis=0)
    else:
        condyle_center = condyle_section['peak_point']

    # Étape 6: Extraire la région finale
    search_radius = 15.0
    distances = np.linalg.norm(side_vertices - condyle_center, axis=1)
    condyle_points = side_vertices[distances < search_radius]

    # Affiner - garder les points supérieurs
    if len(condyle_points) > 50:
        z_threshold = np.percentile(condyle_points[:, 2], 40)
        condyle_points = condyle_points[condyle_points[:, 2] > z_threshold]

    return condyle_points


def fit_ellipsoid_center(points: np.ndarray) -> np.ndarray:
    """
    Estime le centre d'un ellipsoïde englobant les points.
    Utilise une approche simplifiée basée sur les moments.
    """
    # Centre de masse
    centroid = points.mean(axis=0)
    
    # Matrice de covariance pour l'orientation
    centered = points - centroid
    cov = np.cov(centered.T)
    
    # Les valeurs propres donnent les axes principaux
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    return centroid, eigenvalues, eigenvectors


def detect_condyles_precise(mesh: trimesh.Trimesh) -> dict:
    """
    Détection précise des condyles avec isolation des têtes condyliennes.
    """
    vertices = mesh.vertices
    bounds = mesh.bounds
    
    print(f"\n=== Analyse de l'orientation ===")
    print(f"Dimensions: {bounds[1] - bounds[0]}")
    
    # Centre du maillage
    x_center = mesh.centroid[0]
    
    condyles = {}
    
    for side in ["gauche", "droit"]:
        print(f"\n--- Analyse condyle {side} ---")
        
        # Trouver la région du condyle
        condyle_points = find_condyle_region(vertices, side, x_center)
        
        if condyle_points is None or len(condyle_points) < 20:
            print(f"⚠ Impossible de détecter le condyle {side}")
            continue
        
        # Calculer le centre géométrique
        centroid, eigenvalues, eigenvectors = fit_ellipsoid_center(condyle_points)
        
        # Bounding box
        bbox_min = condyle_points.min(axis=0)
        bbox_max = condyle_points.max(axis=0)
        bbox_center = (bbox_min + bbox_max) / 2
        
        # Point le plus haut (pôle supérieur du condyle)
        apex_idx = np.argmax(condyle_points[:, 2])
        apex = condyle_points[apex_idx]
        
        # Point le plus médial et le plus latéral (pour l'axe transversal)
        if side == "gauche":
            medial_idx = np.argmax(condyle_points[:, 0])  # Plus à droite = plus médial
            lateral_idx = np.argmin(condyle_points[:, 0])
        else:
            medial_idx = np.argmin(condyle_points[:, 0])  # Plus à gauche = plus médial
            lateral_idx = np.argmax(condyle_points[:, 0])
        
        medial_pole = condyle_points[medial_idx]
        lateral_pole = condyle_points[lateral_idx]
        
        # Axe principal du condyle (médio-latéral)
        condyle_axis = lateral_pole - medial_pole
        condyle_axis_length = np.linalg.norm(condyle_axis)
        
        condyles[side] = {
            "centroid": centroid.tolist(),
            "bbox_center": bbox_center.tolist(),
            "apex": apex.tolist(),
            "medial_pole": medial_pole.tolist(),
            "lateral_pole": lateral_pole.tolist(),
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "condyle_width": float(condyle_axis_length),
            "num_points": len(condyle_points),
            "principal_axes": eigenvectors.tolist(),
            "axis_lengths": np.sqrt(eigenvalues).tolist(),
            "points": condyle_points
        }
        
        print(f"Centre géométrique: [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]")
        print(f"Apex (pôle supérieur): [{apex[0]:.2f}, {apex[1]:.2f}, {apex[2]:.2f}]")
        print(f"Pôle médial: [{medial_pole[0]:.2f}, {medial_pole[1]:.2f}, {medial_pole[2]:.2f}]")
        print(f"Pôle latéral: [{lateral_pole[0]:.2f}, {lateral_pole[1]:.2f}, {lateral_pole[2]:.2f}]")
        print(f"Largeur du condyle: {condyle_axis_length:.2f} mm")
        print(f"Points analysés: {len(condyle_points)}")
    
    return condyles


def calculate_measurements(condyles: dict) -> dict:
    """Calcule toutes les mesures intercondyliennes."""
    measurements = {}
    
    if "gauche" not in condyles or "droit" not in condyles:
        return measurements
    
    left = condyles["gauche"]
    right = condyles["droit"]
    
    # Distance entre les centres
    left_center = np.array(left["centroid"])
    right_center = np.array(right["centroid"])
    measurements["intercondylar_distance_centers"] = float(np.linalg.norm(right_center - left_center))
    
    # Distance entre les pôles médiaux
    left_medial = np.array(left["medial_pole"])
    right_medial = np.array(right["medial_pole"])
    measurements["intercondylar_distance_medial"] = float(np.linalg.norm(right_medial - left_medial))
    
    # Distance entre les pôles latéraux
    left_lateral = np.array(left["lateral_pole"])
    right_lateral = np.array(right["lateral_pole"])
    measurements["intercondylar_distance_lateral"] = float(np.linalg.norm(right_lateral - left_lateral))
    
    # Axe bicondylien
    axis_vector = right_center - left_center
    measurements["bicondylar_axis"] = {
        "vector": axis_vector.tolist(),
        "midpoint": ((left_center + right_center) / 2).tolist(),
        "angle_to_horizontal": float(np.degrees(np.arcsin(axis_vector[2] / np.linalg.norm(axis_vector))))
    }
    
    # Asymétrie condylienne (différence de hauteur Z)
    measurements["condyle_height_asymmetry"] = float(left_center[2] - right_center[2])
    
    # Asymétrie antéro-postérieure (différence en Y)
    measurements["condyle_ap_asymmetry"] = float(left_center[1] - right_center[1])
    
    return measurements


def export_to_json(condyles: dict, measurements: dict, output_path: str):
    """Exporte les résultats en JSON."""
    export_data = {"condyles": {}, "measurements": measurements}
    
    for side, data in condyles.items():
        export_data["condyles"][side] = {k: v for k, v in data.items() if k != "points"}
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Résultats JSON: {output_path}")


def create_visualization(mesh: trimesh.Trimesh, condyles: dict, output_path: str):
    """Crée un STL avec des marqueurs visuels sur les condyles."""
    meshes = [mesh.copy()]
    
    colors = {"gauche": [255, 0, 0, 255], "droit": [0, 0, 255, 255]}
    
    for side, data in condyles.items():
        # Sphère au centre
        center_sphere = trimesh.creation.icosphere(subdivisions=2, radius=2.5)
        center_sphere.apply_translation(data["centroid"])
        meshes.append(center_sphere)
        
        # Petite sphère à l'apex
        apex_sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.5)
        apex_sphere.apply_translation(data["apex"])
        meshes.append(apex_sphere)
        
        # Marqueurs aux pôles médial et latéral
        for pole in ["medial_pole", "lateral_pole"]:
            pole_sphere = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
            pole_sphere.apply_translation(data[pole])
            meshes.append(pole_sphere)
    
    combined = trimesh.util.concatenate(meshes)
    combined.export(output_path)
    print(f"✓ Visualisation STL: {output_path}")


def create_points_cloud(condyles: dict, output_path: str):
    """Exporte les points des condyles en PLY pour visualisation."""
    all_points = []
    all_colors = []
    
    colors = {"gauche": [255, 100, 100], "droit": [100, 100, 255]}
    
    for side, data in condyles.items():
        points = data["points"]
        all_points.append(points)
        all_colors.extend([colors[side]] * len(points))
    
    all_points = np.vstack(all_points)
    all_colors = np.array(all_colors, dtype=np.uint8)
    
    cloud = trimesh.PointCloud(all_points, colors=all_colors)
    cloud.export(output_path)
    print(f"✓ Nuage de points: {output_path}")


def main(stl_path: str, output_dir: str = "/home/claude"):
    """Fonction principale."""
    print("=" * 60)
    print("DÉTECTION PRÉCISE DES CONDYLES MANDIBULAIRES")
    print("=" * 60)
    
    # Charger
    mesh = load_mandible(stl_path)
    
    # Détecter
    condyles = detect_condyles_precise(mesh)
    
    # Mesures
    print("\n" + "=" * 60)
    print("MESURES INTERCONDYLIENNES")
    print("=" * 60)
    
    measurements = calculate_measurements(condyles)
    
    if measurements:
        print(f"\nDistance intercondylienne (centres): {measurements['intercondylar_distance_centers']:.2f} mm")
        print(f"Distance intercondylienne (pôles médiaux): {measurements['intercondylar_distance_medial']:.2f} mm")
        print(f"Distance intercondylienne (pôles latéraux): {measurements['intercondylar_distance_lateral']:.2f} mm")
        print(f"Asymétrie verticale (G-D): {measurements['condyle_height_asymmetry']:.2f} mm")
        print(f"Asymétrie antéro-postérieure (G-D): {measurements['condyle_ap_asymmetry']:.2f} mm")
    
    # Exports
    base_name = Path(stl_path).stem
    export_to_json(condyles, measurements, f"{output_dir}/{base_name}_condyles_analysis.json")
    create_visualization(mesh, condyles, f"{output_dir}/{base_name}_condyles_marked.stl")
    create_points_cloud(condyles, f"{output_dir}/{base_name}_condyle_points.ply")
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    
    for side in ["gauche", "droit"]:
        if side in condyles:
            c = condyles[side]["centroid"]
            print(f"\nCondyle {side.upper()}:")
            print(f"  Centre: X={c[0]:.2f}, Y={c[1]:.2f}, Z={c[2]:.2f}")
    
    return condyles, measurements


if __name__ == "__main__":
    import sys
    
    stl_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/Mandible_debut.stl"
    main(stl_path)
