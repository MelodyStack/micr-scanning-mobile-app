const {getDefaultConfig, mergeConfig} = require('@react-native/metro-config');

const defaultConfig = getDefaultConfig(__dirname);

/**
 * .tflite is not in Metro's default asset list, so requiring the model would
 * fail with "unable to resolve module". Append rather than replace -- setting
 * assetExts outright drops png, ttf and everything else.
 */
const config = {
  resolver: {
    assetExts: [...defaultConfig.resolver.assetExts, 'tflite'],
  },
};

module.exports = mergeConfig(defaultConfig, config);
