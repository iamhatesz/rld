from flask import Flask, render_template, jsonify

from rld.rollout import Rollout


def init(rollout: Rollout) -> Flask:
    app = Flask(__name__, template_folder="ui", static_folder="ui/static")

    @app.route("/")
    def index():
        return render_template("index.html", episode={})

    @app.route("/trajectories", methods=["GET"])
    def trajectories():
        trajectories = list(range(len(rollout)))
        return jsonify(trajectories=trajectories)

    @app.route("/trajectory/<index>", methods=["GET"])
    def trajectory(index: int):
        raise NotImplementedError

    return app
