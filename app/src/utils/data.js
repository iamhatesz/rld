import _ from 'lodash';

function getActionCodeAttributation(timestep, actionCode) {
  if (actionCode === "picked") {
    return timestep.attributations.picked;
  } else if (_.isInteger(actionCode)) {
    return timestep.attributations.top[actionCode];
  } else {
    throw new Error('Action code should be either a `picked` string or an index.');
  }
}

export {getActionCodeAttributation};
