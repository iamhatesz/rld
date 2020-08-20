import React from 'react';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import ImageEncoder from './utils/imageEncoder';
import {
  attributationPixelColor,
  flattenStackedPixel,
  identityPixelColor,
} from './utils/math';

class ImageAttributationViewer extends React.Component {
  constructor(props) {
    super(props);

    this.obsEncoder = new ImageEncoder(
      this.props.obsWidth,
      this.props.obsHeight,
      (stackedPixel) => identityPixelColor(flattenStackedPixel(stackedPixel))
    );
    this.attrEncoder = new ImageEncoder(
      this.props.obsWidth,
      this.props.obsHeight,
      (pixel) => attributationPixelColor(pixel)
    );
  }

  obsImage() {
    return this.obsEncoder.encode(this.props.currentTimestep.obs);
  }

  attrImage() {
    return this.attrEncoder.encode(this.props.selectedAction.normalized);
  }

  render() {
    return (
      <Container>
        <Row>
          <Col>
            <img src={this.obsImage()} width="300" height="300" />
          </Col>
          <Col>
            <img src={this.attrImage()} width="300" height="300" />
          </Col>
        </Row>
      </Container>
    );
  }
}

ImageAttributationViewer.defaultProps = {
  currentTimestep: null,
  selectedAction: null,
  obsWidth: 84,
  obsHeight: 84,
};

export default ImageAttributationViewer;
