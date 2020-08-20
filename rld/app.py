from dataclasses import asdict, replace

import numpy as np
from flask import Flask, render_template, jsonify, Response
from flask.json import JSONEncoder
from werkzeug.exceptions import NotFound

from rld.exception import TrajectoryNotFound, EndpointNotFound, APIException
from rld.rollout import Rollout


class NumpyJSONEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def init(rollout: Rollout, viewer: str = "none", debug: bool = False) -> Flask:
    this_rollout = rollout
    app = Flask(__name__, template_folder="app", static_folder="app/static")
    app.json_encoder = NumpyJSONEncoder
    if debug:
        from flask_cors import CORS

        CORS(app)

    @app.route("/")
    def index() -> Response:
        return render_template("index.html", viewer=viewer)

    @app.route("/rollout", methods=["GET"])
    def rollout() -> Response:
        return jsonify(
            **asdict(
                replace(
                    this_rollout,
                    trajectories=list(range(len(this_rollout.trajectories))),
                )
            )
        )

    @app.route("/rollout/trajectory/<index>", methods=["GET"])
    def trajectory(index: str) -> Response:
        index = int(index)
        try:
            this_trajectory = this_rollout.trajectories[index]
        except IndexError:
            raise TrajectoryNotFound()
        if viewer == "atari":
            # TODO For Atari we are sending only a few timesteps,
            #  as buffering is not yet implemented
            this_trajectory = replace(
                this_trajectory, timesteps=this_trajectory.timesteps[:10]
            )
        return jsonify(**asdict(this_trajectory))

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
