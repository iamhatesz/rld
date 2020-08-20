import * as THREE from 'three';

function toUint8Color(floatColor) {
  return Math.floor(Math.abs(floatColor) * 255);
}

function toFloatColor(uint8Color) {
  return uint8Color / 255;
}

function flattenStackedPixel(stackedPixel) {
  return stackedPixel[0];
}

function identityPixelColor(value) {
  const normalizedValue = toFloatColor(value);
  return new THREE.Color().setRGB(
    normalizedValue,
    normalizedValue,
    normalizedValue
  );
}

function attributationPixelColor(value) {
  const saturation = 1.0;
  const lightnessAtMin = 0.35;
  const lightnessAtMax = 1.0;
  const huePositive = 0.3;
  const hueNegative = 1.0;

  const lightness =
    lightnessAtMin +
    (1.0 - Math.abs(value)) * (lightnessAtMax - lightnessAtMin);
  const hue = value > 0 ? huePositive : hueNegative;

  return new THREE.Color().setHSL(hue, saturation, lightness);
}

export {
  toUint8Color,
  toFloatColor,
  flattenStackedPixel,
  identityPixelColor,
  attributationPixelColor,
};
