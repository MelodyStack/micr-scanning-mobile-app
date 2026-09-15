module.exports = {
  presets: ['module:@react-native/babel-preset'],
  // Frame processors are worklets: this plugin compiles the 'worklet'-marked
  // functions so they can run on the camera thread instead of the JS thread.
  // Without it the frame processor silently never fires.
  plugins: ['react-native-worklets-core/plugin'],
};
