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


def find_condyle_region(vertices: np.ndarray, side: str, x_center: float, mandible_centroid: np.ndarray) -> np.ndarray:
    """
    Trouve la région du condyle pour un côté donné.

    MÉTHODE: La mandibule a une forme en V.
    Le CONDYLE est à l'extrémité du V = le sommet le plus ÉLOIGNÉ du centroïde.
    La CORONOÏDE est plus proche du centre.

         Condyle G          Condyle D
              \\              //
               \\            //
                \\    ●    //   ← Centroïde
                 \\  /  \\//
                  \\/    \\/
                   Symphyse
    """
    # Séparer par côté
    if side == "gauche":
        side_mask = vertices[:, 0] < x_center
    else:
        side_mask = vertices[:, 0] >= x_center

    side_vertices = vertices[side_mask]

    if len(side_vertices) < 100:
        return None

    # Étape 1: Prendre la région supérieure (branche montante)
    z_threshold = np.percentile(side_vertices[:, 2], 60)
    ramus_region = side_vertices[side_vertices[:, 2] > z_threshold]

    print(f"  → Branche montante: {len(ramus_region)} points")

    # Étape 2: Pour chaque point haut, calculer sa distance au centroïde (en 2D: X,Y)
    # Le condyle est le sommet le plus ÉLOIGNÉ du centroïde
    centroid_2d = mandible_centroid[:2]  # Seulement X, Y

    # Trouver les points les plus hauts (top 20%)
    z_high = np.percentile(ramus_region[:, 2], 80)
    top_region = ramus_region[ramus_region[:, 2] > z_high]

    if len(top_region) < 10:
        z_high = np.percentile(ramus_region[:, 2], 70)
        top_region = ramus_region[ramus_region[:, 2] > z_high]

    # Calculer la distance de chaque point au centroïde (en 2D)
    distances_to_centroid = np.linalg.norm(top_region[:, :2] - centroid_2d, axis=1)

    # Trouver le point le plus éloigné du centroïde parmi les points hauts
    farthest_idx = np.argmax(distances_to_centroid)
    farthest_point = top_region[farthest_idx]
    max_dist = distances_to_centroid[farthest_idx]

    # Trouver aussi le point le plus proche (probablement coronoïde)
    nearest_idx = np.argmin(distances_to_centroid)
    nearest_point = top_region[nearest_idx]
    min_dist = distances_to_centroid[nearest_idx]

    print(f"    Point le plus éloigné du centre: dist={max_dist:.1f}mm (CONDYLE)")
    print(f"    Point le plus proche du centre: dist={min_dist:.1f}mm (coronoïde)")

    # Le condyle = région autour du point le plus éloigné
    condyle_center = farthest_point

    print(f"  ✓ Condyle sélectionné: distance au centre={max_dist:.1f}mm")

    # Étape 3: Extraire la région du condyle
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
    
    # Centre du maillage (centroïde global)
    x_center = mesh.centroid[0]
    mandible_centroid = mesh.centroid  # Centroïde 3D complet

    print(f"Centroïde de la mandibule: {mandible_centroid}")

    condyles = {}

    for side in ["gauche", "droit"]:
        print(f"\n--- Analyse condyle {side} ---")

        # Trouver la région du condyle (basé sur la distance au centroïde)
        condyle_points = find_condyle_region(vertices, side, x_center, mandible_centroid)
        
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
