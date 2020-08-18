import React from 'react';
import Form from "react-bootstrap/Form";
import {Col} from "react-bootstrap";
import ButtonGroup from "react-bootstrap/ButtonGroup";
import DropdownButton from "react-bootstrap/DropdownButton";
import Dropdown from "react-bootstrap/Dropdown";
import Button from "react-bootstrap/Button";
import {MdChevronLeft, MdChevronRight, MdFirstPage, MdLastPage, MdPause, MdPlayArrow} from "react-icons/md/index";
import _ from "lodash";

class Controls extends React.Component {
  isTrajectoryLoaded() {
    return this.props.currentTrajectory !== null;
  }

  isTimestepLoaded() {
    return this.props.currentTimestep !== null;
  }

  isPlaying() {
    return this.props.playing !== null;
  }

  trajectoryLength() {
    if (this.isTrajectoryLoaded()) {
      return this.props.currentTrajectory.timesteps.length;
    } else {
      return 0;
    }
  }

  lastValidTrajectoryIndex() {
    if (this.isTrajectoryLoaded()) {
      return this.trajectoryLength() - 1;
    } else {
      return 0;
    }
  }

  render() {
    return (
      <Form inline>
        <Form.Row>
          <Col>
            <Form.Group controlId="buttons">
              <ButtonGroup aria-label="Control buttons">
                <DropdownButton
                  variant="light"
                  as={ButtonGroup}
                  title={`Trajectory: ${this.props.currentTrajectoryIndex}`}
                >
                  {this.props.trajectories.map((trajectory, index) =>
                    <Dropdown.Item
                      eventKey={index.toString()}
                      key={index}
                      onSelect={eventKey => this.props.fetchTrajectory(_.parseInt(eventKey))}
                    >
                      {index}
                    </Dropdown.Item>
                  )}
                </DropdownButton>
                <Button
                  variant="light"
                  disabled={!this.isTrajectoryLoaded()}
                  onClick={this.props.rewindToBeginning}
                >
                  <MdFirstPage/>
                </Button>
                <Button
                  variant="light"
                  disabled={!this.isTrajectoryLoaded()}
                  onClick={this.props.rewindToPrevious}
                >
                  <MdChevronLeft/>
                </Button>
                <Button
                  variant="light"
                  disabled={!this.isTrajectoryLoaded()}
                  onClick={this.props.togglePlaying}
                >
                  {this.isPlaying() ? <MdPause/> : <MdPlayArrow/>}
                </Button>
                <Button
                  variant="light"
                  disabled={!this.isTrajectoryLoaded()}
                  onClick={this.props.rewindToNext}
                >
                  <MdChevronRight/>
                </Button>
                <Button
                  variant="light"
                  disabled={!this.isTrajectoryLoaded()}
                  onClick={this.props.rewindToEnd}
                >
                  <MdLastPage/>
                </Button>
              </ButtonGroup>
            </Form.Group>
          </Col>
          <Col>
            <Form.Group controlId="timestep">
              <Form.Label column>
                {_.padStart(_.toString(this.props.currentTimestepIndex), 5, "0")}
                /
                {_.padStart(_.toString(this.lastValidTrajectoryIndex()), 5, "0")}
              </Form.Label>
              <Col>
                <Form.Control
                  type="range"
                  max={this.lastValidTrajectoryIndex()}
                  value={this.props.currentTimestepIndex}
                  onChange={(e) => this.props.rewindToPosition(_.toInteger(e.target.value))}
                  disabled={!this.isTrajectoryLoaded()} />
              </Col>
            </Form.Group>
          </Col>
        </Form.Row>
      </Form>
    );
  }
}

Controls.defaultProps = {
  trajectories: [],
  currentTrajectoryIndex: 0,
  currentTrajectory: null,
  currentTimestepIndex: 0,
  currentTimestep: null,
  playing: null,
  fetchTrajectory: null,
  rewindToBeginning: null,
  rewindToPrevious: null,
  rewindToNext: null,
  rewindToEnd: null,
  rewindToPosition: null,
  togglePlaying: null,
};

export default Controls;
