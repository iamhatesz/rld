import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import * as serviceWorker from './serviceWorker';

const rootElement = document.getElementById('root');

let host = rootElement.getAttribute('data-host');
if (host.startsWith('{')) {
  host = 'localhost';
}
let port = rootElement.getAttribute('data-port');
if (port.startsWith('{')) {
  port = 5000;
}
const backendUrl = `http://${host}:${port}/`;
let viewerId = rootElement.getAttribute('data-viewer');
if (viewerId.startsWith('{')) {
  viewerId = 'none';
}

ReactDOM.render(
  <React.StrictMode>
    <App
      backendUrl={backendUrl}
      viewerId={viewerId}
    />
  </React.StrictMode>,
  rootElement
);

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();
