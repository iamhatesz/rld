import React from 'react';
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Form from "react-bootstrap/Form";
import Table from "react-bootstrap/Table";
import flatten from "keypather/flatten";
import _ from "lodash";

class TabularAttributationViewer extends React.Component {
  timestepFeatures() {
    const timestep = this.props.currentTimestep;
    const obs = timestep.obs;
    const raw_attr = this.props.selectedAction.raw;
    const norm_attr = this.props.selectedAction.normalized;
    const labels = Object.entries(flatten(obs)).map((value) => _.first(value));
    return _
      .zip(labels, obs, raw_attr, norm_attr)
      .filter(([label, ...rest]) => label.includes(this.props.filterPhrase));
  }

  render() {
    return (
      <Container fluid>
        <Row>
          <Form className="full-width">
            <Form.Control
              type="text"
              value={this.props.filterPhrase}
              onChange={this.props.filterComponents}
              placeholder="Filter components..."
              size="sm"
              className="full-width"/>
          </Form>
        </Row>
        <Row>
          <Table striped bordered hover>
            <thead>
              <tr>
                <th className="w-20">Component</th>
                <th className="w-20">Label</th>
                <th className="w-15">Value (raw)</th>
                <th className="w-15">Value (real)</th>
                <th className="w-15">Attributation (raw)</th>
                <th className="w-15">Attributation (normalized)</th>
              </tr>
            </thead>
            <tbody>
            {this.timestepFeatures().map(([label, obs, raw_attr, norm_attr]) =>
              <tr key={label}>
                <td>{label}</td>
                <td>n/a</td>
                <td>{obs.toFixed(4)}</td>
                <td>n/a</td>
                <td>{raw_attr.toFixed(6)}</td>
                <td>{norm_attr.toFixed(2)}</td>
              </tr>
            )}
            </tbody>
          </Table>
        </Row>
      </Container>
    );
  }
}

  TabularAttributationViewer.defaultProps = {
    currentTimestep: null,
    selectedAction: null,
  filterPhrase: "",
  filterComponents: null,
};

export default TabularAttributationViewer;
