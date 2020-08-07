import numpy as np
from flask import Flask, render_template, jsonify
from flask.json import JSONEncoder
from flask_cors import CORS
from werkzeug.exceptions import NotFound

from rld.exception import TrajectoryNotFound, EndpointNotFound, APIException
from rld.rollout import Rollout


class NumpyJSONEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def init(rollout: Rollout) -> Flask:
    app = Flask(__name__, template_folder="ui", static_folder="ui/static")
    app.json_encoder = NumpyJSONEncoder
    CORS(app)

    @app.route("/")
    def index():
        return render_template("index.html", episode={})

    @app.route("/trajectories", methods=["GET"])
    def trajectories():
        trajectories = list(range(len(rollout)))
        return jsonify(length=len(trajectories), trajectories=trajectories)

    @app.route("/trajectory/<index>", methods=["GET"])
    def trajectory(index: str):
        index = int(index)
        try:
            this_trajectory = rollout.trajectories[index]
        except IndexError:
            raise TrajectoryNotFound()
        return jsonify(length=len(this_trajectory), timesteps=this_trajectory.timesteps)

    @app.errorhandler(EndpointNotFound)
    @app.errorhandler(TrajectoryNotFound)
    def handle_api_exception(error: APIException):
        response = jsonify(error.as_dict())
        response.status_code = error.status_code
        return response

    @app.errorhandler(NotFound)
    def handle_not_found(error: NotFound):
        return handle_api_exception(EndpointNotFound())

    return app
