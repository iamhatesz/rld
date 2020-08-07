import React from 'react';
import './App.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import Container from "react-bootstrap/Container";
import Navbar from "react-bootstrap/Navbar";
import Nav from "react-bootstrap/Nav";
import Form from "react-bootstrap/Form";
import Table from "react-bootstrap/Table";
import Row from "react-bootstrap/Row";
import _ from "lodash";
import flatten from "keypather/flatten";
import Controls from "./Controls";
import Viewer from "./Viewer";
import Col from "react-bootstrap/Col";
import ListGroup from "react-bootstrap/ListGroup";
import Card from "react-bootstrap/Card";
import DropdownButton from "react-bootstrap/DropdownButton";
import Dropdown from "react-bootstrap/Dropdown";

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      trajectories: [],
      currentTrajectoryIndex: 0,
      currentTrajectory: null,
      currentTimestepIndex: 0,
      currentTimestep: null,
      playing: null,
      filterPhrase: "",
    };
  }

  componentDidMount() {
    this.fetchTrajectoriesList();
  }

  getEndpointUrl(...parts) {
    return this.props.backendUrl + parts.map(elem => _.toString(elem)).join("/");
  }

  fetchTrajectoriesList() {
    fetch(this.getEndpointUrl("trajectories"))
      .then(response => response.json())
      .then(data => this.setState({
        trajectories: data["trajectories"],
      }))
      .then(() => this.fetchTrajectory(0));
  }

  fetchTrajectory(trajectoryIndex) {
    fetch(this.getEndpointUrl("trajectory", trajectoryIndex))
      .then(response => response.json())
      .then(data => this.setState({
        currentTrajectoryIndex: trajectoryIndex,
        currentTrajectory: data,
      }))
      .then(() => this.rewindTo(0));
  }

  isTrajectoryLoaded() {
    return this.state.currentTrajectory !== null;
  }

  isTimestepLoaded() {
    return this.state.currentTimestep !== null;
  }

  isPlaying() {
    return this.state.playing !== null;
  }

  trajectoryLength() {
    if (this.isTrajectoryLoaded()) {
      return this.state.currentTrajectory.length;
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

  rewindTo(timestepIndex) {
    if (timestepIndex < 0 || timestepIndex >= this.trajectoryLength()) {
      return false;
    }
    this.setState({
      currentTimestepIndex: timestepIndex,
      currentTimestep: this.state.currentTrajectory.timesteps[this.state.currentTimestepIndex],
    });
    this.timestepFeatures();
  }

  rewindToBeginning = () => {
    this.pausePlaying();
    this.rewindTo(0);
  }

  rewindToPrevious = () => {
    this.pausePlaying();
    this.rewindTo(this.state.currentTimestepIndex - 1);
  }

  rewindToNext = () => {
    this.pausePlaying();
    this.rewindTo(this.state.currentTimestepIndex + 1);
  }

  rewindToEnd = () => {
    this.pausePlaying();
    this.rewindTo(this.lastValidTrajectoryIndex());
  }

  rewindToPosition = (e) => {
    const timestep = _.toInteger(e.target.value);
    this.pausePlaying();
    this.rewindTo(timestep);
  }

  togglePlaying = () => {
    if (!this.isPlaying()) {
      this.startPlaying();
    } else {
      this.pausePlaying();
    }
  }

  startPlaying = () => {
    this.setState({
      playing: setInterval(this.playTick.bind(this), 500),
    });
  }

  pausePlaying = () => {
    if (this.isPlaying()) {
      clearInterval(this.state.playing);
    }
    this.setState({
      playing: null,
    });
  }

  playTick() {
    this.rewindTo(this.state.currentTimestepIndex + 1);
  }

  filterComponents = (e) => {
    this.setState({
      filterPhrase: e.target.value,
    });
  }

  timestepFeatures() {
    if (!this.isTimestepLoaded()) {
      return [];
    }
    const timestep = this.state.currentTimestep;
    const obs = timestep.obs;
    const attr = timestep.attributations.data;
    const labels = Object.entries(flatten(obs)).map((value) => _.first(value));
    return _.zip(labels, obs, attr).filter(([label, ...rest]) => label.includes(this.state.filterPhrase));
  }

  render() {
    return (
      <div>
        <Navbar bg="light" expand="lg" sticky="top">
          <Navbar.Brand href="/">rld</Navbar.Brand>
          <Navbar.Toggle aria-controls="main-nav"/>
          <Navbar.Collapse id="main-nav">
            <Nav className="mr-auto">
              <Nav.Link href="/viewer">Rollout</Nav.Link>
              <Nav.Link href="/attribution">Observation attributation</Nav.Link>
            </Nav>
          </Navbar.Collapse>
          <Controls
            trajectories={this.state.trajectories}
            currentTrajectoryIndex={this.state.currentTrajectoryIndex}
            currentTrajectory={this.state.currentTrajectory}
            currentTimestepIndex={this.state.currentTimestepIndex}
            currentTimestep={this.state.currentTimestep}
            playing={this.state.playing}
            fetchTrajectory={this.fetchTrajectory.bind(this)}
            rewindToBeginning={this.rewindToBeginning}
            rewindToPrevious={this.rewindToPrevious}
            rewindToNext={this.rewindToNext}
            rewindToEnd={this.rewindToEnd}
            rewindToPosition={this.rewindToPosition}
            togglePlaying={this.togglePlaying}/>
        </Navbar>
        <Container fluid>
          <Row>
            <Col>
              <Row>
                <Form className="full-width">
                  <Form.Control
                    type="text"
                    value={this.state.filterPhrase}
                    onChange={this.filterComponents}
                    placeholder="Filter components..."
                    size="sm"
                    className="full-width"/>
                </Form>
              </Row>
              <Row>
                <Table striped bordered hover>
                  <thead>
                    <tr>
                      <th className="w-25">Component</th>
                      <th className="w-25">Label</th>
                      <th className="w-15">Raw value</th>
                      <th className="w-20">Real value</th>
                      <th className="w-15">Attributation</th>
                    </tr>
                  </thead>
                  <tbody>
                  {this.timestepFeatures().map(([label, obs, attr]) =>
                    <tr key={label}>
                      <td>{label}</td>
                      <td>n/a</td>
                      <td>{obs.toFixed(4)}</td>
                      <td>{obs.toFixed(4)} unit</td>
                      <td>{attr.toFixed(2)}</td>
                    </tr>
                  )}
                  </tbody>
                </Table>
              </Row>
              <Row>
                {this.isTimestepLoaded() ? <Viewer timestep={this.state.currentTimestep}/> : null}
              </Row>
            </Col>
            <Col xs={2} className="bg-light">
              <Container fluid>
                <Row>
                  <DropdownButton variant="secondary" title="Select action" className="full-width">
                    <Dropdown.Item>LEFT</Dropdown.Item>
                    <Dropdown.Item>RIGHT</Dropdown.Item>
                  </DropdownButton>
                </Row>
                <Row>
                  <pre className="action">
                    LEFT
                  </pre>
                </Row>
              </Container>
            </Col>
          </Row>

        </Container>
      </div>
    );
  }
}

App.defaultProps = {
  backendUrl: "http://localhost:5000/",
};

export default App;
