class App {
    constructor(backendUrl) {
        this.backendUrl = backendUrl;

        this.currentTrajectory = null;
        this.currentTimestep = null;
        this.currentIndex = null;
        this.currentTrajectoryLength = null;

        this.playing = null;
        this.fps = 30;

        // UI elements we care about and modify during the scope of the application
        this.playButton = document.getElementById("play");
        this.toStartButton = document.getElementById("to-start");
        this.toEndButton = document.getElementById("to-end");
        this.previousStepButton = document.getElementById("previous-step");
        this.nextStepButton = document.getElementById("next-step");
        this.seekbar = document.getElementById("seekbar");
        this.fpsSelect = document.getElementById("fps");
        this.controls = document.querySelectorAll(".controls button, .controls input[type=range], .controls select");
        this.playlistSelect = document.getElementById("playlist");

        this.trajectoryStep = document.getElementById("trajectory-step");
        this.trajectoryLength = document.getElementById("trajectory-length");

        this.sceneHolder = document.getElementById("scene");

        this.debugObsBox = document.getElementById("debug-obs");
        this.debugAttrBox = document.getElementById("debug-attr");
    }

    run() {
        this.initEventListeners();
        this.fetchTrajectoriesList();
    }

    getEndpointURL(endpoint) {
        return this.backendUrl + endpoint;
    }

    fetchTrajectoriesList() {
        fetch(this.getEndpointURL("/trajectories"))
            .then(validateResponse)
            .then(data => {
                this.clearTrajectoriesList();
                this.populateTrajectoriesList(data["trajectories"]);
            });
    }

    fetchTrajectory(index) {
        this.lockControls();
        this.stopPlaying();
        fetch(this.getEndpointURL("/trajectory/" + index))
            .then(validateResponse)
            .then(data => {
                this.currentTrajectory = data["timesteps"];
                this.currentTrajectoryLength = data["length"];
                this.updateTrajectoryRange(this.currentTrajectoryLength);
                this.moveToTimestep(0);
                this.unlockControls();
            });
    }

    moveToTimestep(index) {
        index = index.clamp(0, this.currentTrajectoryLength - 1);
        this.currentIndex = index;
        this.currentTimestep = this.currentTrajectory[this.currentIndex];
        this.updateTrajectoryProgress(this.currentIndex);
        this.updateDebugWindows(this.currentTimestep["obs"], this.currentTimestep["attributations"]);
    }


    // UI
    initEventListeners() {
        this.playButton.addEventListener("click", event => {
            this.togglePlaying();
            event.preventDefault();
        });
        this.toStartButton.addEventListener("click", event => {
            this.moveToTimestep(0);
            event.preventDefault();
        });
        this.previousStepButton.addEventListener("click", event => {
            this.moveToTimestep(this.currentIndex - 1);
            event.preventDefault();
        });
        this.nextStepButton.addEventListener("click", event => {
            this.moveToTimestep(this.currentIndex + 1);
            event.preventDefault();
        });
        this.toEndButton.addEventListener("click", event => {
            this.moveToTimestep(this.currentTrajectoryLength - 1);
            event.preventDefault();
        });
        this.seekbar.addEventListener("change", event => {
            const index = parseInt(event.target.value);
            this.moveToTimestep(index);
        });
        this.fpsSelect.addEventListener("change", event => {
            const fps = event.target.value;
            this.fps = parseInt(fps);
            const wasPlaying = this.playing != null;
            if (wasPlaying) {
                this.stopPlaying();
                this.startPlaying();
            }
        });
        this.playlistSelect.addEventListener("change", event => {
            const trajectory = event.target.value;
            this.fetchTrajectory(trajectory);
        });
    }

    clearTrajectoriesList() {
        removeAllChildren(this.playlistSelect);
    }

    populateTrajectoriesList(trajectories) {
        trajectories
            .map(index => {
                const option = document.createElement("option");
                option.value = index;
                option.innerText = index;
                this.playlistSelect.appendChild(option);
            });
        // Force an onChange event on the first item in the newly populated trajectories list
        this.playlistSelect.dispatchEvent(new Event("change"));
    }

    updateTrajectoryRange(range) {
        this.trajectoryLength.innerText = range;
        this.seekbar.max = range;
    }

    updateTrajectoryProgress(index) {
        this.trajectoryStep.innerText = index;
        this.seekbar.value = index;
    }

    updateDebugWindows(obs, attr) {
        this.debugObsBox.innerText = JSON.stringify(obs);
        this.debugAttrBox.innerText = JSON.stringify(attr);
    }

    togglePlaying() {
        if (this.playing) {
            this.stopPlaying();
        } else {
            this.startPlaying();
        }
    }

    startPlaying() {
        if (!this.playing) {
            const interval = 1000 / this.fps;
            this.playing = setInterval(this.playTick.bind(this), interval);
        }
    }

    stopPlaying() {
        if (this.playing) {
            clearInterval(this.playing);
        }
        this.playing = null;
    }

    playTick() {
        this.moveToTimestep(this.currentIndex + 1);
        if (this.currentIndex >= (this.currentTrajectoryLength - 1)) {
            this.stopPlaying();
        }
    }

    lockControls() {
        this.controls.forEach(obj => {
            obj.disabled = true;
        });
    }

    unlockControls() {
        this.controls.forEach(obj => {
            obj.disabled = false;
        });
    }
}

Number.prototype.clamp = function (min, max) {
    return Math.max(min, Math.min(this, max));
};

function removeAllChildren(node) {
    while (node.firstChild) {
        node.removeChild(node.lastChild);
    }
}

function validateResponse(response) {
    const contentType = response.headers.get("content-type");
    if (!contentType || !contentType.includes("application/json")) {
        throw new TypeError("Expected response to be in JSON format.")
    }
    return response.json();
}
