from flask import Flask
from routes.ssd import ssd


app = Flask(__name__)


app.register_blueprint(ssd, url_prefix="/api")



if __name__ == '__main__':
    app.run(debug=True, port="5000")
