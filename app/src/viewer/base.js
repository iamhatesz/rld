import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import ImageEncoder from '../utils/imageEncoder';
import { flattenStackedPixel, identityPixelColor } from '../utils/math';

class Viewer {
  domElement() {
    throw new Error('Not implemented');
  }

  attributationViewerType() {
    throw new Error('Not implemented');
  }

  resize(width, height) {
    throw new Error('Not implemented');
  }

  update(timestep) {
    console.warn('Calling update() method, which is not implemented.');
  }

  stringifyAction(action) {
    return JSON.stringify(action);
  }

  iterate(obs, attr) {
    throw new Error('Not implemented');
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

    this.obsImage = document.createElement('img');
    this.obsImage.setAttribute('class', 'obs-image');
    this.obsImage.setAttribute('alt', 'Observation');
    this.obsImage.setAttribute('width', this.width);
    this.obsImage.setAttribute('height', this.height);

    this.encoder = new ImageEncoder(
      this.obsWidth,
      this.obsHeight,
      (stackedPixel) => identityPixelColor(flattenStackedPixel(stackedPixel))
    );
  }

  domElement() {
    return this.obsImage;
  }

  attributationViewerType() {
    return "image";
  }

  resize(width, height) {
    this.width = width;
    this.height = height;
    this.obsImage.setAttribute('width', this.width);
    this.obsImage.setAttribute('height', this.height);
  }

  update(timestep) {
    const obs = timestep.obs;
    this.obsImage.setAttribute('src', this.encoder.encode(obs));
  }

  iterate(obs, attr) {
    const arr = []
    for (let y = 0; y < this.obsHeight; y++) {
      for (let x = 0; x < this.obsWidth; x++) {
        for (let c = 0; c < 4; c++) {
          arr.push({
            label: `pixel at (${y}, ${x}, ${c})`,
            rawValue: obs[y][x][c],
            realValue: obs[y][x][c],
            realValueUnit: '',
            rawAttributation: attr.raw[y][x][c],
            normalizedAttributation: attr.normalized[y][x],
          });
        }
      }
    }
    return arr;
  }
}

class WebGLViewer extends Viewer {
  constructor() {
    super();

    this.width = 600;
    this.height = 400;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(
      75,
      this.width / this.height,
      0.1,
      1000
    );

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
    });
    this.renderer.setSize(this.width, this.height);

    this.light = new THREE.DirectionalLight(0xffffff, 1.5);
    this.light.position.fromArray(this.lightPosition().toArray());
    this.scene.add(this.light);

    this.controls = new OrbitControls(this.camera, this.domElement());
    this.controls.object.position.fromArray(
      this.cameraInitialPosition().toArray()
    );
    this.controls.target = this.centerOfScene();

    this.animate();
  }

  domElement() {
    return this.renderer.domElement;
  }

  attributationViewerType() {
    return "table";
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

export { Viewer, ImageViewer, WebGLViewer };
