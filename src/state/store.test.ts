import { describe, expect, it } from 'vitest';
import { mergeAssets, snapFps, storyOrder } from './store';
import type { Asset } from '@/types/project';

const asset = (id: string, path: string, extra: Partial<Asset> = {}): Asset => ({
  id,
  path,
  name: path.split(/[\\/]/).pop()!,
  kind: 'video',
  durationMs: 15000,
  width: 1280,
  height: 720,
  fps: 24,
  hasAudio: true,
  ...extra,
});

describe('mergeAssets', () => {
  it('treats back- and forward-slash paths as the same file', () => {
    const scanned = [asset('ast_scan1', 'C:\\proj\\cappycat- clips\\clip1.mp4', { order: 0, orderReason: 'clip 1' })];
    const fromPipeline = [asset('ast_pipe1', 'C:/proj/cappycat- clips/clip1.mp4', { sceneTags: ['Bunny'] })];
    const merged = mergeAssets(scanned, fromPipeline);
    expect(merged).toHaveLength(1);
    expect(merged[0].id).toBe('ast_pipe1'); // timeline clips reference the pipeline id
    expect(merged[0].orderReason).toBe('clip 1');
    expect(merged[0].sceneTags).toEqual(['Bunny']);
  });
});

describe('snapFps', () => {
  it('snaps near-standard rates and keeps odd ones', () => {
    expect(snapFps(24.04)).toBe(24);
    expect(snapFps(29.97)).toBe(29.97);
    expect(snapFps(59.9)).toBe(59.94);
    expect(snapFps(12.5)).toBe(12.5);
  });
});

describe('storyOrder', () => {
  it('orders videos by order then natural name, ignoring other kinds', () => {
    const list = [
      asset('a', 'clip10.mp4', { order: 9 }),
      asset('b', 'clip2.mp4', { order: 1 }),
      asset('c', 'look.cube', { kind: 'lut' }),
      asset('d', 'clip1.mp4', { order: 0 }),
    ];
    expect(storyOrder(list).map((a) => a.name)).toEqual(['clip1.mp4', 'clip2.mp4', 'clip10.mp4']);
  });
});
