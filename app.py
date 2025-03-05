from flask import Flask
from src.routes.streaming_routes import streaming_blueprint

app = Flask(__name__)
app.register_blueprint(streaming_blueprint)

if __name__ == '__main__':
    app.run()
