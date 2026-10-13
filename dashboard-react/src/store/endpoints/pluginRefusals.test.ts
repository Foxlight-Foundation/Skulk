import { describe, expect, it } from 'vitest';
import { refusalSentence } from './plugins';

describe('refusalSentence', () => {
  it('shows a plugin\'s next step as written, ending it as a sentence', () => {
    expect(refusalSentence('Add your RunPod API key under Credentials.')).toBe('Add your RunPod API key under Credentials.');
    expect(refusalSentence('  Add your RunPod API key under Credentials  ')).toBe('Add your RunPod API key under Credentials.');
    // Casing is the plugin's: a step may begin with a name.
    expect(refusalSentence('npm must be installed first')).toBe('npm must be installed first.');
    expect(refusalSentence('Is the cleanup computer on?')).toBe('Is the cleanup computer on?');
  });

  it('gives way to the caller\'s guidance for core\'s generic sentence or no reason', () => {
    expect(refusalSentence('configuration refused; reload settings and validation')).toBeNull();
    expect(refusalSentence('   ')).toBeNull();
    expect(refusalSentence(null)).toBeNull();
    expect(refusalSentence(undefined)).toBeNull();
  });
});
