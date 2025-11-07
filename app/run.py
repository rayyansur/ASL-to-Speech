import threading
import webview
import app


def run_flask():
    app.app.run(port=5000, debug=False)


if __name__ == '__main__':
    t = threading.Thread(target=run_flask, daemon=True)
    t.start()
    webview.create_window("ASL Recognition", "http://127.0.0.1:5000", width=600, height=600)
    webview.start()
