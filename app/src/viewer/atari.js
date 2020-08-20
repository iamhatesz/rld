import {ImageViewer} from "./base";

class AtariViewer extends ImageViewer {
  constructor() {
    super(84, 84);
  }

  stringifyAction(action) {
    switch (action) {
      case 0:
        return "NOOP";
      case 1:
        return "FIRE";
      case 2:
        return "UP";
      case 3:
        return "RIGHT";
      case 4:
        return "LEFT";
      case 5:
        return "DOWN";
      case 6:
        return "UPRIGHT";
      case 7:
        return "UPLEFT";
      case 8:
        return "DOWNRIGHT";
      case 9:
        return "DOWNLEFT";
      case 10:
        return "UPFIRE";
      case 11:
        return "RIGHTFIRE";
      case 12:
        return "LEFTFIRE";
      case 13:
        return "DOWNFIRE";
      case 14:
        return "UPRIGHTFIRE";
      case 15:
        return "UPLEFTFIRE";
      case 16:
        return "DOWNRIGHTFIRE";
      case 17:
        return "DOWNLEFTFIRE";
      default:
        return "n/a";
    }
  }
}

export {AtariViewer};
