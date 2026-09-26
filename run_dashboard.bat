@echo off
title Bloodstain Pattern Analysis (BPA) Master Dashboard
echo ==========================================================================
echo  🩸 LAUNCHING BLOODSTAIN PATTERN ANALYSIS (BPA) MASTER DASHBOARD
echo ==========================================================================
echo Opening http://localhost:8501 in default browser...
start http://localhost:8501
python -m streamlit run app.py --server.port 8501
pause
