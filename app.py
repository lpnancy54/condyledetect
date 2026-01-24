#!/usr/bin/env python3
"""
Application SaaS pour l'analyse des condyles mandibulaires.

Permet l'upload d'un fichier STL, l'analyse automatique des condyles,
et l'export des résultats au format JSON et STL marqué.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import trimesh
from flask import Flask, redirect, render_template, request, send_from_directory, url_for

from condyle_detector_v2 import find_condyle_region, fit_ellipsoid_center

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024


def ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def detect_condyles(mesh: trimesh.Trimesh, search_radius: float) -> dict:
    vertices = mesh.vertices
    x_center = mesh.centroid[0]
    condyles: dict = {}

    for side in ["gauche", "droit"]:
        condyle_points = find_condyle_region(
            vertices,
            side,
            x_center,
            search_radius=search_radius,
        )

        if condyle_points is None or len(condyle_points) < 20:
            continue

        centroid, eigenvalues, eigenvectors = fit_ellipsoid_center(condyle_points)
        bbox_min = condyle_points.min(axis=0)
        bbox_max = condyle_points.max(axis=0)
        bbox_center = (bbox_min + bbox_max) / 2

        apex = condyle_points[np.argmax(condyle_points[:, 2])]

        if side == "gauche":
            medial_idx = np.argmax(condyle_points[:, 0])
            lateral_idx = np.argmin(condyle_points[:, 0])
        else:
            medial_idx = np.argmin(condyle_points[:, 0])
            lateral_idx = np.argmax(condyle_points[:, 0])

        medial_pole = condyle_points[medial_idx]
        lateral_pole = condyle_points[lateral_idx]

        condyles[side] = {
            "centroid": centroid,
            "bbox_center": bbox_center,
            "apex": apex,
            "medial_pole": medial_pole,
            "lateral_pole": lateral_pole,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "condyle_width": float(np.linalg.norm(lateral_pole - medial_pole)),
            "num_points": int(len(condyle_points)),
            "principal_axes": eigenvectors,
            "axis_lengths": np.sqrt(eigenvalues),
            "points": condyle_points,
        }

    return condyles


def compute_measurements(condyles: dict) -> dict:
    if "gauche" not in condyles or "droit" not in condyles:
        return {}

    left = condyles["gauche"]
    right = condyles["droit"]

    left_center = left["centroid"]
    right_center = right["centroid"]

    measurements = {
        "intercondylar_distance_centers": float(np.linalg.norm(right_center - left_center)),
        "intercondylar_distance_medial": float(np.linalg.norm(right["medial_pole"] - left["medial_pole"])),
        "intercondylar_distance_lateral": float(np.linalg.norm(right["lateral_pole"] - left["lateral_pole"])),
        "asymetrie_verticale": float(left_center[2] - right_center[2]),
        "asymetrie_ap": float(left_center[1] - right_center[1]),
    }

    axis_vector = right_center - left_center
    axis_length = np.linalg.norm(axis_vector)
    measurements["bicondylar_axis"] = {
        "vector": axis_vector.tolist(),
        "midpoint": ((left_center + right_center) / 2).tolist(),
        "angle_to_horizontal": float(np.degrees(np.arcsin(axis_vector[2] / axis_length)))
        if axis_length > 0
        else 0.0,
    }

    return measurements


def serialize_results(source_file: str, search_radius: float, condyles: dict, measurements: dict) -> dict:
    payload = {
        "source_file": source_file,
        "parameters": {"search_radius": search_radius},
        "condyles": {},
        "measurements": measurements,
    }

    for side, data in condyles.items():
        payload["condyles"][side] = {
            "centroid": data["centroid"].tolist(),
            "bbox_center": data["bbox_center"].tolist(),
            "apex": data["apex"].tolist(),
            "medial_pole": data["medial_pole"].tolist(),
            "lateral_pole": data["lateral_pole"].tolist(),
            "bbox_min": data["bbox_min"].tolist(),
            "bbox_max": data["bbox_max"].tolist(),
            "condyle_width": data["condyle_width"],
            "num_points": data["num_points"],
            "principal_axes": data["principal_axes"].tolist(),
            "axis_lengths": data["axis_lengths"].tolist(),
        }

    return payload


def export_marked_stl(mesh: trimesh.Trimesh, condyles: dict, measurements: dict, output_path: Path) -> None:
    meshes = [mesh.copy()]

    for data in condyles.values():
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=3.0)
        sphere.apply_translation(data["centroid"])
        meshes.append(sphere)

    midpoint = measurements.get("bicondylar_axis", {}).get("midpoint")
    if midpoint is not None:
        mid_sphere = trimesh.creation.icosphere(subdivisions=2, radius=3.5)
        mid_sphere.apply_translation(np.array(midpoint))
        meshes.append(mid_sphere)

    combined = trimesh.util.concatenate(meshes)
    combined.export(output_path)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        ensure_dirs()
        file = request.files.get("stl_file")
        if not file or file.filename == "":
            return render_template("index.html", error="Veuillez sélectionner un fichier STL.")

        search_radius = float(request.form.get("search_radius", 15.0))
        analysis_id = uuid.uuid4().hex
        filename = f"{analysis_id}_{Path(file.filename).name}"
        upload_path = UPLOAD_DIR / filename
        file.save(upload_path)

        mesh = trimesh.load(upload_path)
        condyles = detect_condyles(mesh, search_radius)
        measurements = compute_measurements(condyles)
        payload = serialize_results(str(upload_path), search_radius, condyles, measurements)

        json_path = OUTPUT_DIR / f"{analysis_id}_condyles.json"
        marked_path = OUTPUT_DIR / f"{analysis_id}_marked.stl"

        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        if condyles:
            export_marked_stl(mesh, condyles, measurements, marked_path)

        return redirect(
            url_for(
                "results",
                analysis_id=analysis_id,
                json_file=json_path.name,
                stl_file=marked_path.name if marked_path.exists() else "",
                upload_file=upload_path.name,
            )
        )

    return render_template("index.html")


@app.route("/results")
def results():
    json_file = request.args.get("json_file")
    stl_file = request.args.get("stl_file")
    upload_file = request.args.get("upload_file")
    analysis_id = request.args.get("analysis_id")

    if not json_file:
        return redirect(url_for("index"))

    payload = json.loads((OUTPUT_DIR / json_file).read_text(encoding="utf-8"))
    view_file = stl_file or upload_file or ""
    return render_template(
        "results.html",
        analysis_id=analysis_id,
        payload=payload,
        json_file=json_file,
        stl_file=stl_file,
        view_file=view_file,
    )


@app.route("/download/<path:filename>")
def download(filename: str):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


@app.route("/view/<path:filename>")
def view_file(filename: str):
    for base_dir in (OUTPUT_DIR, UPLOAD_DIR):
        candidate = base_dir / filename
        if candidate.exists():
            return send_from_directory(base_dir, filename, as_attachment=False)
    return ("Fichier introuvable", 404)


if __name__ == "__main__":
    ensure_dirs()
    app.run(host="0.0.0.0", port=5000, debug=True)
