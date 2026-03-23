"""
NeuralScribe v2 — Windows Launcher with Eel
Opens a native browser window. Runs FastAPI server in a background thread.
"""

import sys
import os
import time
import threading


def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"


def start_server():
    import uvicorn
    from backend.api.app import create_app
    app = create_app()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


def wait_for_server(timeout=30):
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"{URL}/api/system/gpu", timeout=2)
            return True
        except:
            time.sleep(0.5)
    return False


def main():
    base = get_base_path()
    os.chdir(base)

    print("=" * 50)
    print("  NeuralScribe v2")
    print("=" * 50)
    print(f"  Starting server on {URL}")
    print()

    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    print("  Waiting for server...")
    if not wait_for_server():
        print("  ERROR: Server failed to start within 30 seconds.")
        print("  Press Enter to close...")
        input()
        return

    print("  Server ready! Opening window...")
    print()
    print("  Close this console window to stop NeuralScribe.")
    print("=" * 50)

    try:
        import eel

        shim_dir = os.path.join(base, "_eel_shim")
        os.makedirs(shim_dir, exist_ok=True)

        shim_html = os.path.join(shim_dir, "index.html")
        with open(shim_html, "w") as f:
            f.write(f'<!DOCTYPE html><html><head><title>NeuralScribe v2</title>'
                    f'<style>body{{font-family:monospace;display:flex;align-items:center;'
                    f'justify-content:center;height:100vh;margin:0}}</style>'
                    f'<script>window.location.href="{URL}";</script>'
                    f'</head><body>Loading NeuralScribe...</body></html>')

        eel.init(shim_dir)

        try:
            eel.start('index.html', mode='chrome', port=0,
                      cmdline_args=[f'--app={URL}', '--disable-extensions',
                                    '--disable-default-apps', '--new-window'],
                      size=(1400, 900))
        except EnvironmentError:
            try:
                edge = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
                if not os.path.exists(edge):
                    edge = r'C:\Program Files\Microsoft\Edge\Application\msedge.exe'
                eel.start('index.html', mode='custom', port=0,
                          cmdline_args=[edge, f'--app={URL}',
                                        '--disable-extensions', '--new-window'],
                          size=(1400, 900))
            except Exception:
                _fallback_browser()

    except ImportError:
        _fallback_browser()


def _fallback_browser():
    import webbrowser
    webbrowser.open(URL)
    print(f"  Opened in default browser: {URL}")
    print("  Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
