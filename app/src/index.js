import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import * as serviceWorker from './serviceWorker';
import _ from 'lodash';

const rootElement = document.getElementById('root');
const host = rootElement.getAttribute('data-host');
const port = _.parseInt(rootElement.getAttribute('data-port'));
const backendUrl = `http://${host}:${port}/`;
const viewerId = rootElement.getAttribute('data-viewer');

ReactDOM.render(
  <React.StrictMode>
    <App
      backendUrl={backendUrl}
      viewerId={viewerId.startsWith('{') ? 'none' : viewerId}
    />
  </React.StrictMode>,
  rootElement
);

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();
