import * as THREE from "three";
import {WebGLViewer} from "./base";

class CartPoleViewer extends WebGLViewer {
  // TODO Make these values compliant with the original CartPole dimensions
  static CART_LENGTH = 3;
  static CART_WIDTH = 1;
  static CART_HEIGHT = 1;
  static POLE_LENGTH = 0.25;
  static POLE_WIDTH = 1.1;
  static POLE_HEIGHT = 3;

  constructor() {
    super();

    this.cart = new THREE.Mesh(
      new THREE.BoxGeometry(CartPoleViewer.CART_LENGTH, CartPoleViewer.CART_HEIGHT, CartPoleViewer.CART_WIDTH),
      new THREE.MeshStandardMaterial({
        color: 0xffffff
      })
    );

    const realPole = new THREE.Mesh(
      new THREE.BoxGeometry(CartPoleViewer.POLE_LENGTH, CartPoleViewer.POLE_HEIGHT, CartPoleViewer.POLE_WIDTH),
      new THREE.MeshStandardMaterial({
        color: 0xffffff
      })
    );
    realPole.position.set(0, CartPoleViewer.POLE_HEIGHT / 2, 0);
    this.pole = new THREE.Group();
    this.pole.position.set(0, 0, 0);
    this.pole.add(realPole);

    this.axle = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-20, 0, 0),
        new THREE.Vector3(20, 0, 0)
      ]),
      new THREE.LineBasicMaterial({
        color: 0xffffff
      })
    );

    this.scene.add(this.cart);
    this.scene.add(this.pole);
    this.scene.add(this.axle);
  }

  update(timestep) {
    const obs = timestep["obs"];
    const cartPosition = obs[0];
    const poleAngle = obs[2];

    this.cart.position.set(cartPosition, 0, 0);
    this.pole.position.set(cartPosition, 0, 0);
    this.pole.rotation.z = poleAngle;
  }

  centerOfScene() {
    return new THREE.Vector3(0, 2, 0);
  }

  cameraInitialPosition() {
    return new THREE.Vector3(0, 3, 5);
  }

  lightPosition() {
    return new THREE.Vector3(0, 5, 10);
  }

  stringifyAction(action) {
    if (action === 0) {
      return "LEFT";
    } else {
      return "RIGHT";
    }
  }
}

export {CartPoleViewer};
