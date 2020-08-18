import React from 'react';
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import TabularAttributationViewer from "./TabularAttributationViewer";
import Container from "react-bootstrap/Container";
import ActionPicker from "./ActionPicker";
import Viewer from "./Viewer";

class AttributationPage extends React.Component {
  render() {
    return (
      <Container fluid>
        <Row>
          <Col>
            <TabularAttributationViewer
              currentTimestep={this.props.currentTimestep}
              selectedAction={this.props.selectedAction}
              filterPhrase={this.props.filterPhrase}
              filterComponents={this.props.filterComponents.bind(this)} />
          </Col>
          <Col xs="2" className="bg-light">
            <Container fluid>
              <Row>
                <ActionPicker
                  currentTimestep={this.props.currentTimestep}
                  selectedAction={this.props.selectedAction}
                  selectPickedAction={this.props.selectPickedAction}
                  selectAction={this.props.selectAction}
                  stringifyAction={this.props.viewer.stringifyAction} />
              </Row>
              <Row>
                <Viewer
                  viewer={this.props.viewer}
                  timestep={this.props.currentTimestep}
                  mode="side"
                />
              </Row>
            </Container>
          </Col>
        </Row>
      </Container>
    );
  }
}

AttributationPage.defaultProps = {
  currentTimestep: null,
  selectedAction: null,
  filterPhrase: "",
  filterComponents: null,
  selectPickedAction: null,
  selectAction: null,
  viewer: null,
};

export default AttributationPage;
