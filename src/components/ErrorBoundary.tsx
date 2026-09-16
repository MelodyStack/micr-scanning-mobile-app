/**
 * Catches render-time errors so a bug shows a readable message instead of a
 * blank screen. Native crashes (TFLite, the camera) still take the process
 * down -- those only show up in `adb logcat`.
 */

import React from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';

interface Props {
  children: React.ReactNode;
}

interface State {
  error: Error | null;
}

export default class ErrorBoundary extends React.Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('[scanner] render failed', error, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) {
      return this.props.children;
    }
    return (
      <View style={styles.root}>
        <Text style={styles.title}>The scanner hit a problem</Text>
        <ScrollView style={styles.scroll}>
          <Text style={styles.message}>{error.message}</Text>
          {!!error.stack && <Text style={styles.stack}>{error.stack}</Text>}
        </ScrollView>
      </View>
    );
  }
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#12151b', padding: 24, paddingTop: 64 },
  title: { color: '#fff', fontSize: 18, fontWeight: '600', marginBottom: 12 },
  scroll: { flex: 1 },
  message: { color: '#f87171', fontSize: 14, marginBottom: 12 },
  stack: { color: '#8b93a1', fontSize: 11, fontFamily: 'monospace' },
});
