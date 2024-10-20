from flask import Flask
from .tracking import init_db

app = Flask(__name__)
init_db()

from app import routes




