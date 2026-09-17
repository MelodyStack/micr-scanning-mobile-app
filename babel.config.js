/**
 * No worklet plugin. Frame processors are gone, and with them
 * react-native-worklets-core -- whose native library fails to link against this
 * React Native version and crashed the app during startup.
 */
module.exports = {
  presets: ['module:@react-native/babel-preset'],
};
