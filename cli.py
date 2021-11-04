from config import HOST, PORT
import sys
import os

def start() -> None:
    """
    This function starts Flask application.
    """
    from app import app
    app.run(debug = True, host = HOST, port = PORT)

if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath(os.getcwd()))
    start()