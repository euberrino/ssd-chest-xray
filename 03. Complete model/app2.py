from flask import Flask
from sqlalchemy.orm import backref, lazyload
from routes.ssd import ssd
import sqlite3



app = Flask(__name__)
db = SQLAlchemy(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

class images(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    path = db.Column(db.String(200),nullable=False)
    width = db.Column(db.Integer,nullable=False)
    height = db.Column(db.Integer, nullable=False)
    thorax = db.Column(db.Boolean,nullable=False)
    thorax_softmax = db.Column(db.Float, nullable=False)
    projection = db.Column(db.String(3))
    pa_softmax = db.Column(db.Float)
    ap_softmax = db.Column(db.Float)
    l_softmax = db.Column(db.Float)
    bb_counts = db.Column(db.Integer)
    bb_cmax = db.Column(db.Float)
    lung_opacity = db.Column(db.Boolean)
    rel = db.relationship('lung_opacity_detections',backref='images',lazyload=True)

class lung_opacity_detections(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    image_id = db.Column(db.Integer,db.ForeignKey('images.id'),nullable=False)
    x_center = db.Column(db.Integer,nullable=False)
    y_center = db.Column(db.Integer, nullable=False)
    relative_width = db.Column(db.Integer,nullable=False)
    relative_height = db.Column(db.Integer, nullable=False)


db.create_all()
app.register_blueprint(ssd, url_prefix="/api")



if __name__ == '__main__':
    # load_dotenv()
    app.run(debug=True, port="5000")
