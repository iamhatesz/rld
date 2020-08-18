import React from 'react';
import Viewer from "./Viewer";
import Container from "react-bootstrap/Container";

class RolloutPage extends React.Component {
  render() {
    return (
      <Container fluid>
        <Viewer
          viewer={this.props.viewer}
          timestep={this.props.currentTimestep}
        />
      </Container>
    );
  }
}

RolloutPage.defaultProps = {
  currentTimestep: null,
  viewer: null,
};

export default RolloutPage;
