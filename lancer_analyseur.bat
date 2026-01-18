@echo off
echo ==========================================
echo  Analyseur de Condyles Mandibulaires
echo ==========================================
echo.

REM Vérifier si Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Python n'est pas installe ou pas dans le PATH
    echo Veuillez installer Python 3.8+ depuis https://python.org
    pause
    exit /b 1
)

REM Installer les dépendances si nécessaire
echo Verification des dependances...
pip install -q numpy trimesh scipy scikit-learn PyQt5 pyqtgraph PyOpenGL

echo.
echo Lancement de l'application...
python condyle_analyzer_gui.py

pause
