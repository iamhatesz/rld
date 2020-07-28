import * as THREE from "./three.module.js";
import { OrbitControls } from "./OrbitControls.js";

class Viewer {
    constructor(width, height) {
        this.width = width;
        this.height = height;

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, this.width / this.height, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({
            antialias: true
        });
        this.renderer.setSize(this.width, this.height);
        this.controls = new OrbitControls(this.camera, this.rendererDOMElement());
        this.controls.object.position.fromArray(this.cameraInitialPosition().toArray());
        this.controls.target = this.centerOfScene();

        this.animate();
    }

    rendererDOMElement() {
        return this.renderer.domElement;
    }

    init() {
        console.warn("Calling init() method, which is not implemented.")
    }

    update(timestep) {
        console.warn("Calling update() method, which is not implemented.")
    }

    stringifyAction(action) {
        return "";
    }

    centerOfScene() {
        return new THREE.Vector3(0, 0, 0);
    }

    cameraInitialPosition() {
        return new THREE.Vector3(5, 5, 5);
    }

    animate() {
        requestAnimationFrame(this.animate.bind(this));
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}

class CartPoleViewer extends Viewer {
    static CART_LENGTH = 3;
    static CART_WIDTH = 1;
    static CART_HEIGHT = 1;
    static POLE_LENGTH = 0.25;
    static POLE_WIDTH = 1.1;
    static POLE_HEIGHT = 3;

    constructor(width, height) {
        super(width, height);

        this.cart = new THREE.Mesh(
            new THREE.BoxGeometry(CartPoleViewer.CART_LENGTH, CartPoleViewer.CART_HEIGHT, CartPoleViewer.CART_WIDTH),
            new THREE.MeshBasicMaterial({
                color: 0x00ff00
            })
        );

        const realPole = new THREE.Mesh(
            new THREE.BoxGeometry(CartPoleViewer.POLE_LENGTH, CartPoleViewer.POLE_HEIGHT, CartPoleViewer.POLE_WIDTH),
            new THREE.MeshBasicMaterial({
                color: 0x00ff00
            })
        );
        realPole.position.set(0, CartPoleViewer.POLE_HEIGHT / 2, 0);
        this.pole = new THREE.Group();
        this.pole.position.set(0, 0, 0);
        this.pole.add(realPole);
    }

    init() {
        this.scene.add(this.cart);
        this.scene.add(this.pole);
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

    stringifyAction(action) {
        if (action === 0) {
            return "LEFT";
        } else {
            return "RIGHT";
        }
    }
}

export { Viewer, CartPoleViewer };
