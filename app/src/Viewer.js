import React from 'react';
import './Viewer.css';
import Form from "react-bootstrap/Form";

class Viewer extends React.Component {
  render() {
    return (
      <div className="viewer viewer-full">
        <div className="viewer-scene">
          <div className="viewer-action">{this.props.timestep.action}</div>
          <div className="viewer-reward">{this.props.timestep.reward}</div>
        </div>
        <div className="viewer-controls">
          <Form>
          </Form>
        </div>
      </div>
    );
  }
}

Viewer.defaultProps = {
  timestep: null,
};

export default Viewer;
