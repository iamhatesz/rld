import * as THREE from "three";
import {OrbitControls} from "three/examples/jsm/controls/OrbitControls";

function floatColorToIntColor(value) {
    return Math.floor(value.clamp(0, 1) * 255);
}

function attributationColor(value, max) {
    const saturation = 1.0;
    const lightnessAtMin = 0.35;
    const lightnessAtMax = 1.0;
    const huePositive = 0.3;
    const hueNegative = 1.0;

    const lightness = (lightnessAtMin + (1.0 - Math.abs(value)) * (lightnessAtMax - lightnessAtMin));
    const hue = value > 0 ? huePositive : hueNegative;

    return [hue, saturation, lightness];
}

class Viewer {
  domElement() {
    throw new Error("Not implemented");
  }

  resize(width, height) {
    throw new Error("Not implemented");
  }

  update(timestep) {
    console.warn("Calling update() method, which is not implemented.");
  }

  stringifyAction(action) {
    return JSON.stringify(action);
  }
}

class ImageViewer extends Viewer {
  /**
   * This Viewer expects observation to be a float array of shape CxHxW, where each value is in range [0; 1].
   */
  constructor(width, height, obsWidth, obsHeight) {
    super();

    this.width = width;
    this.height = height;

    this.obsWidth = obsWidth;
    this.obsHeight = obsHeight;

    this.obsImage = document.createElement("img");
    this.obsImage.setAttribute("class", "obs-image");
    this.obsImage.setAttribute("alt", "Observation");
    this.obsImage.setAttribute("width", this.width);
    this.obsImage.setAttribute("height", this.height);

    this.attrImage = document.createElement("img");
    this.attrImage.setAttribute("class", "attr-image");
    this.attrImage.setAttribute("alt", "Attributation");
    this.attrImage.setAttribute("width", this.width);
    this.attrImage.setAttribute("height", this.height);

    this.container = document.createElement("div")
    this.container.setAttribute("class", "image-container");
    this.container.append(this.obsImage)
    this.container.append(this.attrImage)

    this.canvas = document.createElement("canvas");
    this.canvas.setAttribute("width", this.obsWidth);
    this.canvas.setAttribute("height", this.obsHeight);
    this.ctx = this.canvas.getContext("2d");
    this.buffer = new Uint8ClampedArray(this.obsWidth * this.obsHeight * 4);
    this.imageData = new ImageData(this.buffer, this.obsWidth, this.obsHeight);
  }

  domElement() {
    return this.container;
  }

  update(timestep) {
    const obs = timestep["obs"];
    this.obsImage.setAttribute("src", this.encodedObs(obs));

    const attr = timestep["attributations"]["data"];
    this.attrImage.setAttribute("src", this.encodedAttr(attr));
  }

  encodedObs(obs) {
    for (let y = 0; y < this.obsHeight; y++) {
      for (let x = 0; x < this.obsWidth; x++) {
        const pos = (y * this.obsWidth + x) * 4;
        this.buffer[pos] = floatColorToIntColor(obs[x][y][0]);
        this.buffer[pos + 1] = floatColorToIntColor(obs[x][y][0]);
        this.buffer[pos + 2] = floatColorToIntColor(obs[x][y][0]);
        this.buffer[pos + 3] = 255;
      }
    }

    this.imageData.data.set(this.buffer);
    this.ctx.putImageData(this.imageData, 0, 0);
    return this.canvas.toDataURL();
  }

  encodedAttr(attr) {
    for (let y = 0; y < this.obsHeight; y++) {
      for (let x = 0; x < this.obsWidth; x++) {
        const pos = (y * this.obsWidth + x) * 4;
        const [h, s, l] = attributationColor(attr[x][y], 0.001);
        const pixelColor = new THREE.Color().setHSL(h, s, l);

        this.buffer[pos] = floatColorToIntColor(pixelColor.r);
        this.buffer[pos + 1] = floatColorToIntColor(pixelColor.g);
        this.buffer[pos + 2] = floatColorToIntColor(pixelColor.b);
        this.buffer[pos + 3] = 255;
      }
    }

    this.imageData.data.set(this.buffer);
    this.ctx.putImageData(this.imageData, 0, 0);
    return this.canvas.toDataURL();
  }
}

class WebGLViewer extends Viewer {
  constructor() {
    super();

    this.width = 600;
    this.height = 400;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(75, this.width / this.height, 0.1, 1000);

    this.renderer = new THREE.WebGLRenderer({
      antialias: true
    });
    this.renderer.setSize(this.width, this.height);

    this.light = new THREE.DirectionalLight(0xffffff, 1.5);
    this.light.position.fromArray(this.lightPosition().toArray());
    this.scene.add(this.light);

    this.controls = new OrbitControls(this.camera, this.domElement());
    this.controls.object.position.fromArray(this.cameraInitialPosition().toArray());
    this.controls.target = this.centerOfScene();

    this.animate();
  }

  domElement() {
    return this.renderer.domElement;
  }

  resize(width, height) {
    this.width = width;
    this.height = height;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  centerOfScene() {
    return new THREE.Vector3(0, 0, 0);
  }

  cameraInitialPosition() {
    return new THREE.Vector3(5, 5, 5);
  }

  lightPosition() {
    return new THREE.Vector3(-5, 5, 5);
  }

  animate() {
    requestAnimationFrame(this.animate.bind(this));
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

export {Viewer, ImageViewer, WebGLViewer};
