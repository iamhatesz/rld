class ImageEncoder {
  constructor(width, height, pixelToColor) {
    this.width = width;
    this.height = height;
    this.pixelToColor = pixelToColor;

    this.canvas = document.createElement("canvas");
    this.canvas.setAttribute("width", this.width);
    this.canvas.setAttribute("height", this.height);
    this.ctx = this.canvas.getContext("2d");
    this.buffer = new Uint8ClampedArray(this.width * this.height * 4);
    this.imageData = new ImageData(this.buffer, this.width, this.height);
  }

  encode(array) {
    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        const pos = (x * this.width + y) * 4;
        let pixelColor = this.pixelToColor(array[x][y]);
        this.buffer[pos] = pixelColor.r * 255;
        this.buffer[pos + 1] = pixelColor.g * 255;
        this.buffer[pos + 2] = pixelColor.b * 255;
        this.buffer[pos + 3] = 255;
      }
    }

    this.imageData.data.set(this.buffer);
    this.ctx.putImageData(this.imageData, 0, 0);
    return this.canvas.toDataURL();
  }
}

export default ImageEncoder;
