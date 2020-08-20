import { Viewer } from './base';

class NoneViewer extends Viewer {
  constructor() {
    super();

    this.p = document.createElement('p');
    this.p.innerText = 'No viewer defined.';
  }

  domElement() {
    return this.p;
  }

  resize(width, height) {
    // NOOP
  }

  update(timestep) {
    // NOOP
  }
}

export { NoneViewer };
