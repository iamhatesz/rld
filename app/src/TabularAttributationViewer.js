import React from 'react';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Form from 'react-bootstrap/Form';
import Table from 'react-bootstrap/Table';

class TabularAttributationViewer extends React.Component {
  timestepFeatures() {
    return this.props.iterate(this.props.currentTimestep.obs, this.props.selectedAction)
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
                ({label, rawValue, realValue, rawAttributation, normalizedAttributation}) => (
                  <tr key={label}>
                    <td>{label}</td>
                    <td>{rawValue.toFixed(4)}</td>
                    <td>{realValue.toFixed(4)}</td>
                    <td>{rawAttributation.toFixed(4)}</td>
                    <td>{normalizedAttributation.toFixed(2)}</td>
                  </tr>
                )
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
  filterPhrase: '',
  filterComponents: null,
  viewer: null,
};

export default TabularAttributationViewer;
