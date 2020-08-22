import React from 'react';
import './App.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import Navbar from 'react-bootstrap/Navbar';
import Nav from 'react-bootstrap/Nav';
import _ from 'lodash';
import Controls from './Controls';
import { BrowserRouter as Router, Link, Route, Switch } from 'react-router-dom';
import { CartPoleViewer } from './viewer/cartpole';
import RolloutPage from './RolloutPage';
import AttributationPage from './AttributationPage';
import { AtariViewer } from './viewer/atari';
import Spinner from 'react-bootstrap/Spinner';
import { NoneViewer } from './viewer/none';

const VIEWER_REGISTRY = {
  none: NoneViewer,
  cartpole: CartPoleViewer,
  atari: AtariViewer,
};

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      title: null,
      description: null,
      env: null,
      recordedAt: null,
      trajectories: [],
      currentTrajectoryIndex: 0,
      currentTrajectory: null,
      currentTimestepIndex: 0,
      currentTimestep: null,
      playing: null,
      filterPhrase: '',
      selectedAction: null,
      fetchesInProgress: 0,
    };
    this.viewer = new VIEWER_REGISTRY[this.props.viewerId]();
  }

  componentDidMount() {
    this.fetchTrajectoriesList();
  }

  getEndpointUrl(...parts) {
    return (
      this.props.backendUrl + parts.map((elem) => _.toString(elem)).join('/')
    );
  }

  withLoading(promise) {
    this.setState(
      (prevState) => ({
        fetchesInProgress: prevState.fetchesInProgress + 1,
      }),
      () =>
        promise().then(() =>
          this.setState((prevState) => ({
            fetchesInProgress: prevState.fetchesInProgress - 1,
          }))
        )
    );
  }

  fetchTrajectoriesList() {
    this.withLoading(() =>
      fetch(this.getEndpointUrl('rollout'))
        .then((response) => response.json())
        .then((data) =>
          this.setState({
            title: data['title'],
            description: data['description'],
            env: data['env'],
            recordedAt: data['recorded_at'],
            trajectories: data['trajectories'],
          })
        )
        .then(() => this.fetchTrajectory(0))
    );
  }

  fetchTrajectory(trajectoryIndex) {
    this.withLoading(() =>
      fetch(this.getEndpointUrl('rollout', 'trajectory', trajectoryIndex))
        .then((response) => response.json())
        .then((data) =>
          this.setState({
            currentTrajectoryIndex: trajectoryIndex,
            currentTrajectory: {
              title: data['title'],
              description: data['description'],
              hotspots: data['hotspots'],
              timesteps: data['timesteps'],
            },
          })
        )
        .then(() => this.rewindTo(0))
    );
  }

  isTrajectoryLoaded() {
    return this.state.currentTrajectory !== null;
  }

  isTimestepLoaded() {
    return this.state.currentTimestep !== null;
  }

  isFetchInProgress() {
    return this.state.fetchesInProgress > 0;
  }

  isPlaying() {
    return this.state.playing !== null;
  }

  trajectoryLength() {
    if (this.isTrajectoryLoaded()) {
      return this.state.currentTrajectory.timesteps.length;
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
    this.setState(
      {
        currentTimestepIndex: timestepIndex,
        currentTimestep: this.state.currentTrajectory.timesteps[timestepIndex],
        selectedAction: this.state.currentTrajectory.timesteps[timestepIndex]
          .attributations.picked,
      },
      () => this.viewer.update(this.state.currentTimestep)
    );
  }

  rewindToBeginning = () => {
    this.pausePlaying();
    this.rewindTo(0);
  };

  rewindToPrevious = () => {
    this.pausePlaying();
    this.rewindTo(this.state.currentTimestepIndex - 1);
  };

  rewindToNext = () => {
    this.pausePlaying();
    this.rewindTo(this.state.currentTimestepIndex + 1);
  };

  rewindToEnd = () => {
    this.pausePlaying();
    this.rewindTo(this.lastValidTrajectoryIndex());
  };

  rewindToPosition = (index) => {
    this.pausePlaying();
    this.rewindTo(index);
  };

  togglePlaying = () => {
    if (!this.isPlaying()) {
      this.startPlaying();
    } else {
      this.pausePlaying();
    }
  };

  startPlaying = () => {
    this.setState({
      playing: setInterval(this.playTick.bind(this), 50),
    });
  };

  pausePlaying = () => {
    if (this.isPlaying()) {
      clearInterval(this.state.playing);
    }
    this.setState({
      playing: null,
    });
  };

  playTick() {
    this.rewindTo(this.state.currentTimestepIndex + 1);
  }

  filterComponents = (e) => {
    this.setState({
      filterPhrase: e.target.value,
    });
  };

  selectPickedAction = (e) => {
    this.setState({
      selectedAction: this.state.currentTimestep.attributations.picked,
    });
  };

  selectAction(actionId) {
    this.setState({
      selectedAction: this.state.currentTimestep.attributations.top[actionId],
    });
  }

  render() {
    return (
      <Router>
        {this.isFetchInProgress() && (
          <div className="loader">
            <Spinner animation="grow" variant="primary" />
          </div>
        )}
        <Navbar bg="light" expand="lg" sticky="top">
          <Navbar.Brand href="/">rld</Navbar.Brand>
          <Navbar.Toggle aria-controls="main-nav" />
          <Navbar.Collapse id="main-nav">
            <Nav className="mr-auto">
              <Nav.Link as={Link} to="/">
                Viewer
              </Nav.Link>
              <Nav.Link as={Link} to="/attributation">
                Attributation
              </Nav.Link>
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
            togglePlaying={this.togglePlaying}
          />
        </Navbar>
        <Switch>
          <Route path="/" exact>
            {this.isTimestepLoaded() && (
              <RolloutPage
                currentTimestep={this.state.currentTimestep}
                viewer={this.viewer}
              />
            )}
          </Route>
          <Route path="/attributation" exact>
            {this.isTimestepLoaded() && (
              <AttributationPage
                currentTimestep={this.state.currentTimestep}
                selectedAction={this.state.selectedAction}
                filterPhrase={this.state.filterPhrase}
                filterComponents={this.filterComponents}
                selectPickedAction={this.selectPickedAction}
                selectAction={this.selectAction.bind(this)}
                viewer={this.viewer}
                viewerId={this.props.viewerId}
              />
            )}
          </Route>
        </Switch>
      </Router>
    );
  }
}

App.defaultProps = {
  backendUrl: 'http://localhost:5000/',
  viewerId: 'atari',
};

export default App;
