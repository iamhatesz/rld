import React from 'react';
import Row from 'react-bootstrap/Row';
import DropdownButton from 'react-bootstrap/DropdownButton';
import Dropdown from 'react-bootstrap/Dropdown';
import Container from 'react-bootstrap/Container';
import _ from 'lodash';
import './ActionPicker.css';

class ActionPicker extends React.Component {
  render() {
    const {attributations: {picked: pickedAction, top: topActions}} = this.props.currentTimestep;

    return (
      <Container fluid>
        <Row>
          <DropdownButton
            variant="secondary"
            title="Select action"
            className="action-picker"
          >
            <Dropdown.Header>Picked action</Dropdown.Header>
            <Dropdown.Item onClick={this.props.selectPickedAction}>
              {this.props.stringifyAction(pickedAction.action)}(
              {pickedAction.prob.toFixed(2)})
            </Dropdown.Item>
            <Dropdown.Divider />
            <Dropdown.Header>Top actions</Dropdown.Header>
            {topActions.map((attr, index) => (
              <Dropdown.Item
                eventKey={index.toString()}
                key={index}
                onSelect={(eventKey) =>
                  this.props.selectAction(_.parseInt(eventKey))
                }
              >
                {this.props.stringifyAction(attr.action)} (
                {attr.prob.toFixed(2)})
              </Dropdown.Item>
            ))}
          </DropdownButton>
        </Row>
        <Row>
          <pre className="selected-action">
            {this.props.stringifyAction(this.props.selectedAction.action)}
          </pre>
        </Row>
      </Container>
    );
  }
}

ActionPicker.defaultProps = {
  currentTimestep: null,
  selectedAction: null,
  selectPickedAction: null,
  selectAction: null,
  stringifyAction: null,
};

export default ActionPicker;
