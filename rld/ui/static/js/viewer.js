import * as THREE from "./three.module.js";
import { OrbitControls } from "./OrbitControls.js";

Number.prototype.clamp = function (min, max) {
    return Math.max(min, Math.min(this, max));
};

function attributationColor(value, max) {
    const saturation = 1.0;
    const lightnessAtMin = 0.35;
    const lightnessAtMax = 1.0;
    const huePositive = 0.3;
    const hueNegative = 1.0;

    const lightness = (lightnessAtMin + (1.0 - (Math.abs(value) / max).clamp(0, max)) * (lightnessAtMax - lightnessAtMin));
    const hue = value > 0 ? huePositive : hueNegative;

    return [hue, saturation, lightness];
}

function floatColorToIntColor(value) {
    return Math.floor(value.clamp(0, 1) * 255);
}

class Viewer {
    constructor(width, height) {
        this.width = width;
        this.height = height;
    }

    domElement() {
        throw new Error("Not implemented");
    }

    init() {
        console.warn("Calling init() method, which is not implemented.");
    }

    update(timestep) {
        console.warn("Calling update() method, which is not implemented.");
    }

    stringifyAction(action) {
        return "";
    }
}

class ImageViewer extends Viewer {
    /**
     * This Viewer expects observation to be a float array of shape CxHxW, where each value is in range [0; 1].
     */
    constructor(width, height, obsWidth, obsHeight, obsChannels, frameStackingInChannelDim = false) {
        super(width, height);

        this.obsWidth = obsWidth;
        this.obsHeight = obsHeight;
        this.obsChannels = obsChannels;
        this.frameStackingInChannelDim = frameStackingInChannelDim;

        this.image = document.createElement("img");
        this.image.setAttribute("alt", "Observation");
        this.image.setAttribute("width", this.width);
        this.image.setAttribute("height", this.height);

        this.canvas = document.createElement("canvas");
        this.canvas.setAttribute("width", this.obsWidth);
        this.canvas.setAttribute("height", this.obsHeight);
        this.ctx = this.canvas.getContext("2d");
        this.buffer = new Uint8ClampedArray(this.obsWidth * this.obsHeight * 4);
        this.imageData = new ImageData(this.buffer, this.obsWidth, this.obsHeight);
    }

    domElement() {
        return this.image;
    }

    update(timestep) {
        const obs = timestep["obs"];

        for (let y = 0; y < this.obsHeight; y++) {
            for (let x = 0; x < this.obsWidth; x++) {
                const pos = (y * this.obsWidth + x) * 4;
                if (this.frameStackingInChannelDim) {
                    // We are using first frame in a stack and treating this as a greyscale value
                    this.buffer[pos] = floatColorToIntColor(obs[x][y][0]);
                    this.buffer[pos + 1] = floatColorToIntColor(obs[x][y][0]);
                    this.buffer[pos + 2] = floatColorToIntColor(obs[x][y][0]);
                    this.buffer[pos + 3] = 255;
                } else {
                    // We are assuming RGB coding here
                    this.buffer[pos] = floatColorToIntColor(obs[x][y][0]);
                    this.buffer[pos + 1] = floatColorToIntColor(obs[x][y][1]);
                    this.buffer[pos + 2] = floatColorToIntColor(obs[x][y][2]);
                    this.buffer[pos + 3] = 255;
                }
            }
        }

        this.imageData.data.set(this.buffer);
        this.ctx.putImageData(this.imageData, 0, 0);

        this.image.setAttribute("src", this.canvas.toDataURL());
    }
}

class WebGLViewer extends Viewer {
    constructor(width, height) {
        super(width, height);

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

class CartPoleViewer extends WebGLViewer {
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
    }

    init() {
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

        if (timestep["attributations"] !== null) {
            const attr = timestep["attributations"]["data"];
            const cartTotal = attr[0] + attr[1];
            const poleTotal = attr[2] + attr[3];
            const [cartH, cartS, cartL] = attributationColor(cartTotal, 1);
            const [poleH, poleS, poleL] = attributationColor(poleTotal, 1);
            this.cart.material.color.setHSL(cartH, cartS, cartL);
            this.pole.children[0].material.color.setHSL(poleH, poleS, poleL);
        }
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

export { ImageViewer, CartPoleViewer };
