from flask import Flask
from flask_cors import CORS
from flask_session import Session

from paths import api
from session import init_session

# TO DO: добавить waitress для production сервера.
app = Flask(__name__)
app.config.from_pyfile('config.py')
Session(app)
CORS(app, supports_credentials=True)
api.init_app(app)


@app.before_request
def before_request_func():
    init_session()


if __name__ == "__main__":
    port = app.config.get('FLASK_PORT')

    if app.config.get('DEBUG'):
        app.run(host='0.0.0.0', port=port)
    else:
        app.run()
