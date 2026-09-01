import { describe, expect, it } from 'vitest';
import { darkTheme, lightTheme } from './theme';

describe('topology theme parity with skulk-app', () => {
  it('uses the Den node palette with an opaque route-occluding surface', () => {
    expect(darkTheme.colors).toMatchObject({
      topologyNodeSurface: '#151D34',
      topologyNodeMemory: '#93AEDF',
      topologyNodeComputeTrack: 'rgba(147, 174, 223, 0.22)',
      topologyNodeCompute: '#F2A03D',
      topologyNodeSelection: '#92A4C2',
      topologyNodeText: '#F4F6FB',
      topologyNodeLabel: 'rgba(147, 174, 223, 0.75)',
      topologyNodeDetail: 'rgba(232, 237, 247, 0.56)',
      topologyNodeHealthy: '#54C79A',
      topologyNodeSyncing: '#93AEDF',
      topologyNodeWarning: '#F2A03D',
      topologyNodeDanger: '#F2707E',
      topologyNodeDotBorder: 'rgba(43, 58, 99, 0.28)',
    });
  });

  it('uses the native Noon Ridge node palette in light mode', () => {
    expect(lightTheme.colors).toMatchObject({
      topologyNodeSurface: '#FFFFFF',
      topologyNodeMemory: '#456FB0',
      topologyNodeComputeTrack: 'rgba(17, 33, 60, 0.16)',
      topologyNodeCompute: '#AC580A',
      topologyNodeSelection: '#52657F',
      topologyNodeText: '#11213C',
      topologyNodeLabel: '#5F7086',
      topologyNodeDetail: '#65707E',
      topologyNodeHealthy: '#1C7A54',
      topologyNodeSyncing: '#456FB0',
      topologyNodeWarning: '#96601A',
      topologyNodeDanger: '#B23A44',
      topologyNodeDotBorder: '#FFFFFF',
    });
  });
});
