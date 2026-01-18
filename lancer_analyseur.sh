#!/bin/bash
echo "=========================================="
echo " Analyseur de Condyles Mandibulaires"
echo "=========================================="
echo

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo "ERREUR: Python 3 n'est pas installé"
    echo "Installez-le avec: sudo apt install python3 python3-pip"
    exit 1
fi

# Installer les dépendances si nécessaire
echo "Vérification des dépendances..."
pip3 install -q numpy trimesh scipy scikit-learn PyQt5 pyqtgraph PyOpenGL

echo
echo "Lancement de l'application..."
python3 condyle_analyzer_gui.py
