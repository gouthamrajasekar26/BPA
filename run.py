"""
PROJECT LAUNCHER SCRIPT
=======================
Starts the Streamlit BPA Master Forensic Dashboard server and opens it automatically
in your default web browser.
"""

import os
import sys
import subprocess
import time
import webbrowser

# Ensure Windows terminal standard output handles UTF-8 correctly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    print("==========================================================================")
    print(" [BPA] LAUNCHING BLOODSTAIN PATTERN ANALYSIS (BPA) PHASE 1 DASHBOARD")
    print("==========================================================================")
    
    # 1. Open Browser Tab
    url = "http://localhost:8501"
    print(f"[Launcher] Opening web browser at {url}...")
    
    # Delay opening slightly so server initializes
    def open_browser():
        time.sleep(2.0)
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"[Launcher] Could not launch browser automatically: {e}")
            
    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # 2. Run Streamlit Server
    print("[Launcher] Starting Streamlit server on port 8501...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port=8501"], check=True)
    except KeyboardInterrupt:
        print("\n[Launcher] Dashboard server stopped.")
    except Exception as e:
        print(f"[Launcher] Error running Streamlit server: {e}")

if __name__ == "__main__":
    main()
