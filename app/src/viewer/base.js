import * as THREE from "three";
import {OrbitControls} from "three/examples/jsm/controls/OrbitControls";
import _ from "lodash";
import ImageEncoder from "../utils/imageEncoder";
import {flattenStackedPixel, identityPixelColor} from "../utils/math";


function floatColorToIntColor(value) {
    return Math.floor(_.clamp(value, 0, 1) * 255);
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
  constructor(obsWidth, obsHeight) {
    super();

    // These are just initial dummy values
    // - the real numbers will be set on the first call to resize()
    this.width = 100;
    this.height = 100;

    this.obsWidth = obsWidth;
    this.obsHeight = obsHeight;

    this.obsImage = document.createElement("img");
    this.obsImage.setAttribute("class", "obs-image");
    this.obsImage.setAttribute("alt", "Observation");
    this.obsImage.setAttribute("width", this.width);
    this.obsImage.setAttribute("height", this.height);

    this.encoder = new ImageEncoder(
      this.obsWidth,
      this.obsHeight,
      (stackedPixel) => identityPixelColor(
        flattenStackedPixel(stackedPixel)
      )
    );
  }

  domElement() {
    return this.obsImage;
  }

  resize(width, height) {
    this.width = width;
    this.height = height;
    this.obsImage.setAttribute("width", this.width);
    this.obsImage.setAttribute("height", this.height);
  }

  update(timestep) {
    const obs = timestep.obs;
    this.obsImage.setAttribute("src", this.encoder.encode(obs));
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
