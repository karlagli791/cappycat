import { useMemo, useState } from 'react';
import { storyOrder, useEditor } from '@/state/store';
import { api } from '@/lib/tauri';
import { fmtDuration, naturalCompare } from '@/lib/format';
import type { Asset } from '@/types/project';
import { FolderIcon, WarnIcon } from './Icons';
import { confirmAction } from './Dialogs';

interface Props {
  onImport: () => void;
  onDropFiles: (paths: string[]) => void;
  onScanFolder: (pick: boolean) => void;
}

type SortMode = 'story' | 'name' | 'duration' | 'tags';

export default function AssetPool({ onImport, onDropFiles, onScanFolder }: Props) {
  const assets = useEditor((s) => s.project.assets);
  const selection = useEditor((s) => s.selection);
  const select = useEditor((s) => s.select);
  const addClip = useEditor((s) => s.addClipFromAsset);
  const removeAssetNow = useEditor((s) => s.removeAsset);
  const tracks = useEditor((s) => s.project.tracks);
  const removeAsset = async (a: Asset) => {
    const uses = tracks.reduce((n, t) => n + t.clips.filter((c) => c.assetId === a.id).length, 0);
    const ok = await confirmAction(
      `Remove "${a.name}"?`,
      uses ? `It is used by ${uses} clip${uses === 1 ? '' : 's'} on the timeline; they are deleted too (undo with Ctrl+Z).` : 'It is removed from the media library (the file on disk is kept).',
      'Remove',
      true,
    );
    if (ok) removeAssetNow(a.id);
  };
  const reorderAsset = useEditor((s) => s.reorderAsset);
  const serverUrl = useEditor((s) => s.mediaServerUrl);
  const clipsFolder = useEditor((s) => s.clipsFolder);
  const warnings = useEditor((s) => s.clipsWarnings);
  const [sort, setSort] = useState<SortMode>('story');
  const [over, setOver] = useState(false);

  const { videos, others } = useMemo(() => {
    const vids = storyOrder(assets);
    const rest = assets.filter((a) => a.kind !== 'video');
    let list = vids;
    if (sort === 'name') list = [...vids].sort((a, b) => naturalCompare(a.name, b.name));
    if (sort === 'duration') list = [...vids].sort((a, b) => b.durationMs - a.durationMs);
    if (sort === 'tags') list = [...vids].sort((a, b) => naturalCompare((a.sceneTags ?? []).join(','), (b.sceneTags ?? []).join(',')));
    return { videos: list, others: rest.sort((a, b) => naturalCompare(a.name, b.name)) };
  }, [assets, sort]);

  const storyIndex = useMemo(() => new Map(storyOrder(assets).map((a, i) => [a.id, i])), [assets]);
  const totalMs = videos.reduce((m, a) => m + a.durationMs, 0);

  return (
    <>
      <div className="panel-title">
        <span>Media library</span>
        <span style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
          <select value={sort} onChange={(e) => setSort(e.target.value as SortMode)} style={{ fontSize: 11, padding: '1px 4px' }}>
            <option value="story">Story order</option>
            <option value="name">Name</option>
            <option value="duration">Duration</option>
            <option value="tags">Scene tags</option>
          </select>
          <button className="small primary nowrap" onClick={onImport} title="Import media (Ctrl+I)">
            Import
          </button>
        </span>
      </div>
      <div
        className="panel-body"
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          const paths: string[] = [];
          for (const f of Array.from(e.dataTransfer.files)) {
            const p = (f as File & { path?: string }).path;
            if (p) paths.push(p);
          }
          if (paths.length) onDropFiles(paths);
        }}
      >
        <div className="folder-bar">
          <div className="folder-path" title={clipsFolder ?? ''}>
            <FolderIcon /> {clipsFolder ? clipsFolder.split(/[\\/]/).slice(-2).join('/') : 'clips folder not loaded'}
          </div>
          <button className="small" onClick={() => onScanFolder(false)} title="Re-read the clips folder and its filename order">
            Rescan
          </button>
          <button className="small ghost" onClick={() => onScanFolder(true)} title="Choose a different folder">
            …
          </button>
        </div>
        {warnings.length ? (
          <div className="order-warnings">
            {warnings.map((w) => (
              <div key={w} className="warn-line">
                <WarnIcon /> {w}
              </div>
            ))}
          </div>
        ) : null}

        <div className="asset-list">
          {videos.map((a) => (
            <AssetRow
              key={a.id}
              asset={a}
              index={storyIndex.get(a.id) ?? null}
              canMove={sort === 'story'}
              isFirst={storyIndex.get(a.id) === 0}
              isLast={storyIndex.get(a.id) === videos.length - 1}
              serverUrl={serverUrl}
              selected={selection.assetId === a.id}
              onSelect={() => select([], a.id)}
              onAdd={() => addClip(a.id)}
              onRemove={() => void removeAsset(a)}
              onMove={(d) => reorderAsset(a.id, d)}
            />
          ))}
          {others.length ? <div className="asset-sep">Audio, images & LUTs</div> : null}
          {others.map((a) => (
            <AssetRow
              key={a.id}
              asset={a}
              index={null}
              canMove={false}
              isFirst
              isLast
              serverUrl={serverUrl}
              selected={selection.assetId === a.id}
              onSelect={() => select([], a.id)}
              onAdd={() => addClip(a.id)}
              onRemove={() => void removeAsset(a)}
              onMove={() => undefined}
            />
          ))}
        </div>
        <div className={`dropzone ${over ? 'over' : ''}`} onClick={onImport}>
          Drop raw 15–50 s AI clips or .cube LUTs here
          <div className="hint" style={{ marginTop: 4 }}>
            {videos.length} clips · {fmtDuration(totalMs)} of footage
          </div>
        </div>
      </div>
    </>
  );
}

function AssetRow({
  asset,
  index,
  canMove,
  isFirst,
  isLast,
  serverUrl,
  selected,
  onSelect,
  onAdd,
  onRemove,
  onMove,
}: {
  asset: Asset;
  index: number | null;
  canMove: boolean;
  isFirst: boolean;
  isLast: boolean;
  serverUrl: string;
  selected: boolean;
  onSelect: () => void;
  onAdd: () => void;
  onRemove: () => void;
  onMove: (dir: -1 | 1) => void;
}) {
  const thumb = asset.kind === 'video' && serverUrl ? api.frameUrl(serverUrl, asset.path, Math.min(1000, asset.durationMs / 2), 160) : '';
  return (
    <div
      className={`asset ${selected ? 'selected' : ''}`}
      onClick={onSelect}
      onDoubleClick={onAdd}
      draggable
      onDragStart={(e) => {
        e.dataTransfer.setData('application/x-cappycat-asset', asset.id);
        e.dataTransfer.effectAllowed = 'copy';
      }}
      title={asset.orderReason ? `Position ${index != null ? index + 1 : '?'}: ${asset.orderReason}\nDouble-click or drag to the timeline` : 'Double-click or drag to the timeline'}
    >
      <div className="thumb" style={{ position: 'relative' }}>
        {thumb ? <img src={thumb} alt="" loading="lazy" /> : asset.kind === 'lut' ? 'LUT' : asset.kind === 'audio' ? '♪' : 'VIDEO'}
        {index != null ? <span className="order-badge">{index + 1}</span> : null}
      </div>
      <div style={{ minWidth: 0 }}>
        <div className="name">{asset.name}</div>
        <div className="meta">
          {asset.kind === 'lut' ? '.cube 3D LUT' : `${fmtDuration(asset.durationMs)} · ${asset.width}×${asset.height} · ${Math.round(asset.fps)}fps`}
        </div>
        {asset.orderReason && index != null ? <div className="order-reason">↳ {asset.orderReason}</div> : null}
        {asset.sceneTags?.length ? (
          <div className="tag-list" style={{ marginTop: 3 }} title="Characters recognised in this clip">
            {asset.sceneTags.map((t) => (
              <span key={t} className="tag cast-tag">
                {t}
              </span>
            ))}
          </div>
        ) : null}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {canMove ? (
          <>
            <button className="small ghost" disabled={isFirst} onClick={(e) => { e.stopPropagation(); onMove(-1); }} title="Move earlier">
              ▲
            </button>
            <button className="small ghost" disabled={isLast} onClick={(e) => { e.stopPropagation(); onMove(1); }} title="Move later">
              ▼
            </button>
          </>
        ) : null}
        {asset.kind !== 'lut' && !canMove ? (
          <button className="small" onClick={(e) => { e.stopPropagation(); onAdd(); }} title="Append to timeline">
            +
          </button>
        ) : null}
        <button className="small ghost" onClick={(e) => { e.stopPropagation(); onRemove(); }} title="Remove from project">
          ×
        </button>
      </div>
    </div>
  );
}
