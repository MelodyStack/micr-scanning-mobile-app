import React from 'react';
import { StatusBar } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import ErrorBoundary from './src/components/ErrorBoundary';
import ScanScreen from './src/screens/ScanScreen';

export default function App() {
  return (
    <SafeAreaProvider>
      <StatusBar barStyle="light-content" />
      <ErrorBoundary>
        <ScanScreen />
      </ErrorBoundary>
    </SafeAreaProvider>
  );
}
