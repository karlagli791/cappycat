/** Monochrome inline icons (currentColor), replacing colour emoji. */
type P = { size?: number };

export function LockIcon({ size = 12 }: P) {
  return (
    <svg width={size} height={size} viewBox="0 0 12 12" aria-hidden="true">
      <rect x="2" y="5.5" width="8" height="5.5" rx="1" fill="currentColor" />
      <path d="M3.8 5.5V4a2.2 2.2 0 0 1 4.4 0v1.5" fill="none" stroke="currentColor" strokeWidth="1.3" />
    </svg>
  );
}

export function UnlockIcon({ size = 12 }: P) {
  return (
    <svg width={size} height={size} viewBox="0 0 12 12" aria-hidden="true">
      <rect x="2" y="5.5" width="8" height="5.5" rx="1" fill="none" stroke="currentColor" strokeWidth="1.2" />
      <path d="M3.8 5.5V4a2.2 2.2 0 0 1 4.3-.7" fill="none" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  );
}

export function EyeIcon({ size = 12, off = false }: P & { off?: boolean }) {
  return (
    <svg width={size} height={size} viewBox="0 0 12 12" aria-hidden="true">
      <path d="M1 6s1.8-3.5 5-3.5S11 6 11 6s-1.8 3.5-5 3.5S1 6 1 6z" fill="none" stroke="currentColor" strokeWidth="1.1" />
      <circle cx="6" cy="6" r="1.6" fill="currentColor" />
      {off ? <path d="M1.5 10.5l9-9" stroke="currentColor" strokeWidth="1.3" /> : null}
    </svg>
  );
}

export function SpeakerIcon({ size = 12, off = false }: P & { off?: boolean }) {
  return (
    <svg width={size} height={size} viewBox="0 0 12 12" aria-hidden="true">
      <path d="M1.5 4.5h2l3-2.5v8l-3-2.5h-2z" fill="currentColor" />
      {off ? (
        <path d="M8 4.5l3 3M11 4.5l-3 3" stroke="currentColor" strokeWidth="1.2" />
      ) : (
        <path d="M8.2 4.2a2.6 2.6 0 0 1 0 3.6M9.6 3a4.3 4.3 0 0 1 0 6" fill="none" stroke="currentColor" strokeWidth="1.1" />
      )}
    </svg>
  );
}

export function FolderIcon({ size = 12 }: P) {
  return (
    <svg width={size} height={size} viewBox="0 0 12 12" aria-hidden="true">
      <path d="M1 3a1 1 0 0 1 1-1h2.6l1 1.2H10a1 1 0 0 1 1 1V9.5a1 1 0 0 1-1 1H2a1 1 0 0 1-1-1z" fill="none" stroke="currentColor" strokeWidth="1.1" />
    </svg>
  );
}

export function WarnIcon({ size = 12 }: P) {
  return (
    <svg width={size} height={size} viewBox="0 0 12 12" aria-hidden="true">
      <path d="M6 1.2l5 9.3H1z" fill="none" stroke="currentColor" strokeWidth="1.1" strokeLinejoin="round" />
      <path d="M6 4.6v2.8M6 8.6v.6" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  );
}

export function FitIcon({ size = 12 }: P) {
  return (
    <svg width={size} height={size} viewBox="0 0 12 12" aria-hidden="true">
      <path d="M1 4V1h3M8 1h3v3M11 8v3H8M4 11H1V8" fill="none" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  );
}
