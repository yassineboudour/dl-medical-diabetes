@echo off
title DL Medical — Streamlit App
echo.
echo ========================================================
echo   Deep Learning Medical — EMSI Casablanca 2025-2026
echo   Application Streamlit
echo ========================================================
echo.

:: Aller dans le dossier du projet
cd /d "c:\Users\yassi\OneDrive\Bureau\DL projet diabetes"

echo [1/2] Verification de Streamlit...
py -m streamlit --version
if errorlevel 1 (
    echo Installation de Streamlit...
    py -m pip install streamlit plotly shap lime -q
)

echo.
echo [2/2] Lancement de l'application sur http://localhost:8501
echo.
echo    Appuyez sur Ctrl+C pour arreter l'application.
echo.
py -m streamlit run app.py --server.port 8501

pause
