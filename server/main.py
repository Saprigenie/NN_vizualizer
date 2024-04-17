from flask import Flask
from flask_cors import CORS
from flask_session import Session
from waitress import serve

from paths import api
from session import init_session
from utility import remove_folder_content

# Удаляем прошлые сессии перед стартом сервера.
remove_folder_content("flask_session")

app = Flask(__name__)
app.config.from_pyfile("config.py")
Session(app)
CORS(app, supports_credentials=True, expose_headers=["Content-Disposition"])
api.init_app(app)


@app.before_request
def before_request_func():
    init_session()


if __name__ == "__main__":
    port = app.config.get("FLASK_PORT")

    if app.config.get("DEBUG"):
        app.run(host="0.0.0.0", port=port)
    else:
        serve(
            app,
            host="0.0.0.0",
            port=port,
            connection_limit=100,
            cleanup_interval=5,
            channel_timeout=30,
        )
