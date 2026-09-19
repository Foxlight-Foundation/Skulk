import { describe, expect, it } from 'vitest';
import { darkTheme, lightTheme } from './theme';

describe('Studio Night topology semantics', () => {
  it('uses the Night node palette with an opaque route-occluding surface', () => {
    expect(darkTheme.colors).toMatchObject({
      topologyNodeSurface: '#131a33',
      topologyNodeMemory: '#93aedf',
      topologyNodeComputeTrack: 'rgba(147,174,223,.18)',
      topologyNodeCompute: '#54c79a',
      topologyNodeSelection: '#93aedf',
      topologyNodeText: '#e8edf7',
      topologyNodeLabel: '#8a9ab8',
      topologyNodeDetail: '#6c7ea3',
      topologyNodeHealthy: '#54c79a',
      topologyNodeSyncing: '#93aedf',
      topologyNodeWarning: '#f2a03d',
      topologyNodeDanger: '#e5655f',
      topologyNodeDotBorder: '#2b3a63',
    });
  });

  it('uses the Noon Ridge node palette in light mode', () => {
    expect(lightTheme.colors).toMatchObject({
      topologyNodeSurface: '#f5f8fc',
      topologyNodeMemory: '#4d7cc4',
      topologyNodeComputeTrack: 'rgba(17,33,60,.16)',
      topologyNodeCompute: '#1c7a54',
      topologyNodeSelection: '#4d7cc4',
      topologyNodeText: '#11213c',
      topologyNodeLabel: '#5f7086',
      topologyNodeDetail: '#7a8aa3',
      topologyNodeHealthy: '#1c7a54',
      topologyNodeSyncing: '#4d7cc4',
      topologyNodeWarning: '#b35c0a',
      topologyNodeDanger: '#b23a44',
      topologyNodeDotBorder: '#c9d9f0',
    });
  });
});
