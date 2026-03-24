import type { DeviceModel } from '../../types/topology';

const APPLE_LOGO_PATH =
  'M788.1 340.9c-5.8 4.5-108.2 62.2-108.2 190.5 0 148.4 130.3 200.9 134.2 202.2-.6 3.2-20.7 71.9-68.7 141.9-42.8 61.6-87.5 123.1-155.5 123.1s-85.5-39.5-164-39.5c-76.5 0-103.7 40.8-165.9 40.8s-105.6-57-155.5-127C46.7 790.7 0 663 0 541.8c0-194.4 126.4-297.5 250.8-297.5 66.1 0 121.2 43.4 162.7 43.4 39.5 0 101.1-46 176.3-46 28.5 0 130.9 2.6 198.3 99.2zm-234-181.5c31.1-36.9 53.1-88.1 53.1-139.3 0-7.1-.6-14.3-1.9-20.1-50.6 1.9-110.8 33.7-147.1 75.8-28.5 32.4-55.1 83.6-55.1 135.5 0 7.8 1.3 15.6 1.9 18.1 3.2.6 8.4 1.3 13.6 1.3 45.4 0 102.5-30.4 135.5-71.3z';
const LOGO_NATIVE_WIDTH = 814;
const LOGO_NATIVE_HEIGHT = 1000;

export interface DeviceIconProps {
  model: DeviceModel;
  /** 0-100 */
  ramPercent: number;
  width: number;
  height: number;
  wireColor?: string;
  strokeWidth?: number;
  fillColor?: string;
  /** unique id prefix for SVG clip paths */
  clipId?: string;
}

export function DeviceIcon({
  model,
  ramPercent,
  width,
  height,
  wireColor = 'rgba(179,179,179,0.8)',
  strokeWidth = 1.5,
  fillColor = 'rgba(255,215,0,0.08)',
  clipId = 'device',
}: DeviceIconProps) {
  const cx = width / 2;
  const cy = height / 2;

  if (model === 'mac-studio') {
    return <MacStudio cx={cx} cy={cy} width={width} height={height} ramPercent={ramPercent} wireColor={wireColor} strokeWidth={strokeWidth} clipId={clipId} />;
  }
  if (model === 'mac-mini') {
    return <MacMini cx={cx} cy={cy} width={width} height={height} ramPercent={ramPercent} wireColor={wireColor} strokeWidth={strokeWidth} clipId={clipId} />;
  }
  if (model === 'macbook-pro') {
    return <MacBookPro cx={cx} cy={cy} width={width} height={height} ramPercent={ramPercent} wireColor={wireColor} strokeWidth={strokeWidth} clipId={clipId} />;
  }
  return <HexagonDefault cx={cx} cy={cy} width={width} height={height} wireColor={wireColor} strokeWidth={strokeWidth} fillColor={fillColor} ramPercent={ramPercent} clipId={clipId} />;
}

interface MacStudioProps {
  cx: number; cy: number; width: number; height: number;
  ramPercent: number; wireColor: string; strokeWidth: number; clipId: string;
}

function MacStudio({ cx, cy, width, height, ramPercent, wireColor, strokeWidth, clipId }: MacStudioProps) {
  const boxW = width * 0.82;
  const boxH = height * 0.7;
  const x = cx - boxW / 2;
  const y = cy - boxH / 2;
  const cornerRadius = 4;
  const topSurfaceH = boxH * 0.15;
  const bodyH = boxH - topSurfaceH;
  const memFillH = (ramPercent / 100) * bodyH;
  const slotH = boxH * 0.14;
  const vSlotW = boxW * 0.05;
  const vSlotY = y + topSurfaceH + bodyH * 0.6;

  return (
    <g>
      <defs>
        <clipPath id={`${clipId}-studio`}>
          <rect x={x} y={y + topSurfaceH} width={boxW} height={bodyH} rx={cornerRadius - 1} />
        </clipPath>
      </defs>
      {/* Body */}
      <rect x={x} y={y} width={boxW} height={boxH} rx={cornerRadius}
        fill="#1a1a1a" stroke={wireColor} strokeWidth={strokeWidth} />
      {/* RAM fill */}
      {ramPercent > 0 && (
        <rect x={x} y={y + topSurfaceH + (bodyH - memFillH)} width={boxW} height={memFillH}
          fill="rgba(255,215,0,0.75)" clipPath={`url(#${clipId}-studio)`} />
      )}
      {/* USB-C slots */}
      {[cx - boxW * 0.28, cx - boxW * 0.18].map((vx, i) => (
        <rect key={i} x={vx - vSlotW / 2} y={vSlotY} width={vSlotW} height={slotH}
          fill="rgba(0,0,0,0.35)" rx={1.5} />
      ))}
      {/* SD card slot */}
      <rect x={cx - boxW * 0.11} y={vSlotY} width={boxW * 0.12} height={slotH * 0.6}
        fill="rgba(0,0,0,0.35)" rx={1} />
    </g>
  );
}

interface MacMiniProps {
  cx: number; cy: number; width: number; height: number;
  ramPercent: number; wireColor: string; strokeWidth: number; clipId: string;
}

function MacMini({ cx, cy, width, height, ramPercent, wireColor, strokeWidth, clipId }: MacMiniProps) {
  const boxW = width * 0.85;
  const boxH = height * 0.58;
  const x = cx - boxW / 2;
  const y = cy - boxH / 2;
  const cornerRadius = 3;
  const topSurfaceH = boxH * 0.2;
  const bodyH = boxH - topSurfaceH;
  const memFillH = (ramPercent / 100) * bodyH;
  const slotH = boxH * 0.2;
  const vSlotW = boxW * 0.045;
  const vSlotY = y + topSurfaceH + bodyH * 0.45;

  return (
    <g>
      <defs>
        <clipPath id={`${clipId}-mini`}>
          <rect x={x} y={y + topSurfaceH} width={boxW} height={bodyH} rx={cornerRadius - 1} />
        </clipPath>
      </defs>
      <rect x={x} y={y} width={boxW} height={boxH} rx={cornerRadius}
        fill="#1a1a1a" stroke={wireColor} strokeWidth={strokeWidth} />
      {ramPercent > 0 && (
        <rect x={x} y={y + topSurfaceH + (bodyH - memFillH)} width={boxW} height={memFillH}
          fill="rgba(255,215,0,0.75)" clipPath={`url(#${clipId}-mini)`} />
      )}
      {[cx - boxW * 0.24, cx - boxW * 0.14].map((vx, i) => (
        <rect key={i} x={vx - vSlotW / 2} y={vSlotY} width={vSlotW} height={slotH}
          fill="rgba(0,0,0,0.35)" rx={1.2} />
      ))}
    </g>
  );
}

interface MacBookProProps {
  cx: number; cy: number; width: number; height: number;
  ramPercent: number; wireColor: string; strokeWidth: number; clipId: string;
}

function MacBookPro({ cx, cy, width, height, ramPercent, wireColor, strokeWidth, clipId }: MacBookProProps) {
  const screenW = width * 0.72;
  const screenH = height * 0.62;
  const baseH = height * 0.26;
  const screenX = cx - screenW / 2;
  const screenY = cy - (screenH + baseH) / 2;
  const bezel = 3;
  const innerW = screenW - bezel * 2;
  const innerH = screenH - bezel * 2;
  const memFillH = (ramPercent / 100) * innerH;

  // Apple logo sizing
  const logoTargetH = screenH * 0.22;
  const logoScale = logoTargetH / LOGO_NATIVE_HEIGHT;
  const logoX = cx - (LOGO_NATIVE_WIDTH * logoScale) / 2;
  const logoY = screenY + screenH / 2 - logoTargetH / 2;

  // Base (keyboard) trapezoid
  const baseY = screenY + screenH;
  const baseTopW = screenW;
  const baseBottomW = width * 0.85;
  const baseTopX = cx - baseTopW / 2;
  const baseBottomX = cx - baseBottomW / 2;

  const keyboardX = baseTopX + 6;
  const keyboardY = baseY + 3;
  const keyboardW = baseTopW - 12;
  const keyboardH = baseH * 0.55;

  const trackpadW = baseTopW * 0.4;
  const trackpadX = cx - trackpadW / 2;
  const trackpadY = baseY + keyboardH + 5;
  const trackpadH = baseH * 0.3;

  return (
    <g>
      <defs>
        <clipPath id={`${clipId}-mbp-screen`}>
          <rect x={screenX + bezel} y={screenY + bezel} width={innerW} height={innerH} rx={2} />
        </clipPath>
      </defs>
      {/* Screen frame */}
      <rect x={screenX} y={screenY} width={screenW} height={screenH} rx={3}
        fill="#1a1a1a" stroke={wireColor} strokeWidth={strokeWidth} />
      {/* Screen inner */}
      <rect x={screenX + bezel} y={screenY + bezel} width={innerW} height={innerH} rx={2}
        fill="#0a0a12" />
      {/* RAM fill on screen */}
      {ramPercent > 0 && (
        <rect x={screenX + bezel} y={screenY + bezel + (innerH - memFillH)} width={innerW} height={memFillH}
          fill="rgba(255,215,0,0.85)" clipPath={`url(#${clipId}-mbp-screen)`} />
      )}
      {/* Apple logo */}
      <path d={APPLE_LOGO_PATH}
        transform={`translate(${logoX}, ${logoY}) scale(${logoScale})`}
        fill="#FFFFFF" opacity={0.9} />
      {/* Keyboard base */}
      <path
        d={`M ${baseTopX} ${baseY} L ${baseTopX + baseTopW} ${baseY} L ${baseBottomX + baseBottomW} ${baseY + baseH} L ${baseBottomX} ${baseY + baseH} Z`}
        fill="#2c2c2c" stroke={wireColor} strokeWidth={1} />
      {/* Keyboard area */}
      <rect x={keyboardX} y={keyboardY} width={keyboardW} height={keyboardH}
        fill="rgba(0,0,0,0.2)" rx={2} />
      {/* Trackpad */}
      <rect x={trackpadX} y={trackpadY} width={trackpadW} height={trackpadH}
        fill="rgba(255,255,255,0.08)" rx={2} />
    </g>
  );
}

interface HexagonDefaultProps {
  cx: number; cy: number; width: number; height: number;
  wireColor: string; strokeWidth: number; fillColor: string;
  ramPercent: number; clipId: string;
}

function HexagonDefault({ cx, cy, width, height, wireColor, strokeWidth, fillColor, ramPercent, clipId }: HexagonDefaultProps) {
  const hexRadius = Math.min(width, height) * 0.42;
  const points = Array.from({ length: 6 }, (_, i) => {
    const angle = ((i * 60 - 30) * Math.PI) / 180;
    return `${cx + hexRadius * Math.cos(angle)},${cy + hexRadius * Math.sin(angle)}`;
  }).join(' ');

  // Bounding box of the hexagon for RAM fill clipping
  const hexTop = cy - hexRadius;
  const hexH = hexRadius * 2;
  const hexLeft = cx - hexRadius;
  const hexW = hexRadius * 2;
  const memFillH = (ramPercent / 100) * hexH;

  return (
    <g>
      <defs>
        <clipPath id={`${clipId}-hex`}>
          <polygon points={points} />
        </clipPath>
      </defs>
      <polygon points={points} fill="#1a1a1a" stroke={wireColor} strokeWidth={strokeWidth} />
      {ramPercent > 0 && (
        <rect x={hexLeft} y={hexTop + (hexH - memFillH)} width={hexW} height={memFillH}
          fill="rgba(255,215,0,0.75)" clipPath={`url(#${clipId}-hex)`} />
      )}
    </g>
  );
}
