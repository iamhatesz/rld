import React from 'react';
import './Viewer.css';

class Viewer extends React.Component {
  constructor(props) {
    super(props);
    this.sceneRef = React.createRef();
  }

  componentDidMount() {
    if (this.isFullMode()) {
      this.props.viewer.resize(600, 400);
    } else {
      this.props.viewer.resize(
        this.sceneRef.current.offsetWidth,
        this.sceneRef.current.offsetHeight
      );
    }
    this.sceneRef.current.appendChild(this.props.viewer.domElement());
  }

  componentWillUnmount() {
    this.sceneRef.current.removeChild(this.props.viewer.domElement());
  }

  isFullMode() {
    return this.props.mode === 'full';
  }

  render() {
    return (
      <div
        className={`viewer ${
          this.isFullMode() ? 'viewer-full' : 'viewer-side'
        }`}
      >
        <div className="viewer-scene" ref={this.sceneRef}>
          {this.isFullMode() && (
            <>
              <div className="viewer-action">
                {this.props.viewer.stringifyAction(this.props.timestep.action)}
              </div>
              <div className="viewer-reward">{this.props.timestep.reward}</div>
            </>
          )}
        </div>
      </div>
    );
  }
}

Viewer.defaultProps = {
  viewer: null,
  timestep: null,
  mode: 'full', // Available modes: "full" and "side"
};

export default Viewer;
