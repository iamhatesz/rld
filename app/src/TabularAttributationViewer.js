import React from 'react';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Form from 'react-bootstrap/Form';
import Table from 'react-bootstrap/Table';

class TabularAttributationViewer extends React.Component {
  timestepFeatures() {
    const selectedActionAttr = this.props.selectedAction === "picked"
      ? this.props.currentTimestep.attributations.picked
      : this.props.currentTimestep.attributations.top[this.props.selectedAction];
    return this.props.viewer.iterate(this.props.currentTimestep.obs, selectedActionAttr)
      .filter(
        ({label, ...rest}) => label.includes(this.props.filterPhrase)
      )
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
              className="full-width"
            />
          </Form>
        </Row>
        <Row>
          <Table striped bordered hover>
            <thead>
              <tr>
                <th className="w-20">Label</th>
                <th className="w-20">Value (raw)</th>
                <th className="w-20">Value (real)</th>
                <th className="w-20">Attributation (raw)</th>
                <th className="w-20">Attributation (normalized)</th>
              </tr>
            </thead>
            <tbody>
              {this.timestepFeatures().map(
                ({label, rawValue, realValue, realValueUnit, rawAttributation, normalizedAttributation}) => (
                  <tr key={label}>
                    <td>{label}</td>
                    <td>{rawValue.toFixed(4)}</td>
                    <td>{realValue.toFixed(4)} {realValueUnit}</td>
                    <td>{this.formatAttributation(rawAttributation, 4)}</td>
                    <td>{this.formatAttributation(normalizedAttributation, 2)}</td>
                  </tr>
                )
              )}
            </tbody>
          </Table>
        </Row>
      </Container>
    );
  }

  formatAttributation(attr, fractionDigits = 4) {
    if (Array.isArray(attr)) {
      const joined = attr.map(a => this.formatAttributation(a, fractionDigits)).join(', ');
      return `[${joined}]`;
    } else {
      return attr.toFixed(fractionDigits);
    }
  }
}

TabularAttributationViewer.defaultProps = {
  currentTimestep: null,
  selectedAction: null,
  filterPhrase: '',
  filterComponents: null,
  viewer: null,
};

export default TabularAttributationViewer;
