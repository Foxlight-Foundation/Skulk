/**
 * DeviceIcon
 *
 * Renders an SVG <g> element representing a Mac device.
 * Must be placed inside an <svg> parent.
 * Ported directly from DeviceIcon.svelte.
 */
import React from 'react';

const APPLE_LOGO_PATH =
  'M788.1 340.9c-5.8 4.5-108.2 62.2-108.2 190.5 0 148.4 130.3 200.9 134.2 202.2-.6 3.2-20.7 71.9-68.7 141.9-42.8 61.6-87.5 123.1-155.5 123.1s-85.5-39.5-164-39.5c-76.5 0-103.7 40.8-165.9 40.8s-105.6-57-155.5-127C46.7 790.7 0 663 0 541.8c0-194.4 126.4-297.5 250.8-297.5 66.1 0 121.2 43.4 162.7 43.4 39.5 0 101.1-46 176.3-46 28.5 0 130.9 2.6 198.3 99.2zm-234-181.5c31.1-36.9 53.1-88.1 53.1-139.3 0-7.1-.6-14.3-1.9-20.1-50.6 1.9-110.8 33.7-147.1 75.8-28.5 32.4-55.1 83.6-55.1 135.5 0 7.8 1.3 15.6 1.9 18.1 3.2.6 8.4 1.3 13.6 1.3 45.4 0 102.5-30.4 135.5-71.3z';
const LOGO_NATIVE_HEIGHT = 1000;
const LOGO_NATIVE_WIDTH = 814;

const WIRE = 'rgba(179,179,179,0.8)';
const STROKE_W = 1.5;

export interface DeviceIconProps {
  deviceType: string;
  cx: number;
  cy: number;
  size?: number;
  ramPercent?: number;
  uid?: string;
}

export const DeviceIcon: React.FC<DeviceIconProps> = ({
  deviceType,
  cx,
  cy,
  size = 60,
  ramPercent = 60,
  uid = 'dev',
}) => {
  const modelLower = deviceType.toLowerCase();
  const isMacStudio = modelLower === 'mac studio' || modelLower === 'mac mini';

  // ── Mac Studio / Mac Mini geometry ──────────────────────────────────────────
  const stW = size * 1.25;
  const stH = size * 0.85;
  const stX = cx - stW / 2;
  const stY = cy - stH / 2;
  const stCorner = 4;
  const stTopH = stH * 0.15;
  const stSlotH = stH * 0.14;
  const stVSlotW = stW * 0.05;
  const stVSlotY = stY + stTopH + (stH - stTopH) * 0.6;
  const stVSlot1X = stX + stW * 0.18;
  const stVSlot2X = stX + stW * 0.28;
  const stHSlotW = stW * 0.2;
  const stHSlotX = stX + stW * 0.5 - stHSlotW / 2;
  const stMemTotalH = stH - stTopH;
  const stMemH = (ramPercent / 100) * stMemTotalH;
  const studioClipId = `di-studio-${uid}`;

  // ── MacBook Pro geometry ────────────────────────────────────────────────────
  const mbW = (size * 1.6 * 0.85) / 1.15;
  const mbH = size * 0.85;
  const mbX = cx - mbW / 2;
  const mbY = cy - mbH / 2;
  const mbScreenH = mbH * 0.7;
  const mbBaseH = mbH * 0.3;
  const mbScreenW = mbW * 0.85;
  const mbScreenX = cx - mbScreenW / 2;
  const mbBezel = 3;
  const mbMemTotalH = mbScreenH - mbBezel * 2;
  const mbMemH = (ramPercent / 100) * mbMemTotalH;
  const mbLogoTargetH = mbScreenH * 0.22;
  const mbLogoScale = mbLogoTargetH / LOGO_NATIVE_HEIGHT;
  const mbLogoX = cx - (LOGO_NATIVE_WIDTH * mbLogoScale) / 2;
  const mbLogoY = mbY + mbScreenH / 2 - (LOGO_NATIVE_HEIGHT * mbLogoScale) / 2;
  const mbBaseY = mbY + mbScreenH;
  const mbBaseTopW = mbScreenW;
  const mbBaseBottomW = mbW;
  const mbBaseTopX = cx - mbBaseTopW / 2;
  const mbBaseBottomX = cx - mbBaseBottomW / 2;
  const mbKbX = mbBaseTopX + 6;
  const mbKbY = mbBaseY + 3;
  const mbKbW = mbBaseTopW - 12;
  const mbKbH = mbBaseH * 0.55;
  const mbTpW = mbBaseTopW * 0.4;
  const mbTpX = cx - mbTpW / 2;
  const mbTpY = mbBaseY + mbKbH + 5;
  const mbTpH = mbBaseH * 0.3;
  const screenClipId = `di-screen-${uid}`;

  if (isMacStudio) {
    return (
      <g>
        <defs>
          <clipPath id={studioClipId}>
            <rect x={stX} y={stY + stTopH} width={stW} height={stH - stTopH} rx={stCorner - 1} />
          </clipPath>
        </defs>
        {/* Main body */}
        <rect x={stX} y={stY} width={stW} height={stH} rx={stCorner} fill="#1a1a1a" stroke={WIRE} strokeWidth={STROKE_W} />
        {/* Memory fill */}
        {ramPercent > 0 && (
          <rect
            x={stX}
            y={stY + stTopH + (stMemTotalH - stMemH)}
            width={stW}
            height={stMemH}
            fill="rgba(255,215,0,0.75)"
            clipPath={`url(#${studioClipId})`}
          />
        )}
        {/* Top surface divider */}
        <line x1={stX} y1={stY + stTopH} x2={stX + stW} y2={stY + stTopH} stroke="rgba(179,179,179,0.3)" strokeWidth="0.5" />
        {/* Vertical slots */}
        <rect x={stVSlot1X - stVSlotW / 2} y={stVSlotY} width={stVSlotW} height={stSlotH} fill="rgba(0,0,0,0.35)" rx="1.5" />
        <rect x={stVSlot2X - stVSlotW / 2} y={stVSlotY} width={stVSlotW} height={stSlotH} fill="rgba(0,0,0,0.35)" rx="1.5" />
        {/* Horizontal slot */}
        <rect x={stHSlotX} y={stVSlotY} width={stHSlotW} height={stSlotH * 0.6} fill="rgba(0,0,0,0.35)" rx="1" />
      </g>
    );
  }

  return (
    <g>
      <defs>
        <clipPath id={screenClipId}>
          <rect x={mbScreenX + mbBezel} y={mbY + mbBezel} width={mbScreenW - mbBezel * 2} height={mbScreenH - mbBezel * 2} rx="2" />
        </clipPath>
      </defs>
      {/* Screen frame */}
      <rect x={mbScreenX} y={mbY} width={mbScreenW} height={mbScreenH} rx="3" fill="#1a1a1a" stroke={WIRE} strokeWidth={STROKE_W} />
      {/* Screen inner */}
      <rect x={mbScreenX + mbBezel} y={mbY + mbBezel} width={mbScreenW - mbBezel * 2} height={mbScreenH - mbBezel * 2} rx="2" fill="#0a0a12" />
      {/* Memory fill */}
      {ramPercent > 0 && (
        <rect
          x={mbScreenX + mbBezel}
          y={mbY + mbBezel + (mbMemTotalH - mbMemH)}
          width={mbScreenW - mbBezel * 2}
          height={mbMemH}
          fill="rgba(255,215,0,0.85)"
          clipPath={`url(#${screenClipId})`}
        />
      )}
      {/* Apple logo */}
      <path
        d={APPLE_LOGO_PATH}
        transform={`translate(${mbLogoX}, ${mbLogoY}) scale(${mbLogoScale})`}
        fill="#FFFFFF"
        opacity="0.9"
      />
      {/* Keyboard base (trapezoidal) */}
      <path
        d={`M ${mbBaseTopX} ${mbBaseY} L ${mbBaseTopX + mbBaseTopW} ${mbBaseY} L ${mbBaseBottomX + mbBaseBottomW} ${mbBaseY + mbBaseH} L ${mbBaseBottomX} ${mbBaseY + mbBaseH} Z`}
        fill="#2c2c2c"
        stroke={WIRE}
        strokeWidth="1"
      />
      {/* Keyboard area */}
      <rect x={mbKbX} y={mbKbY} width={mbKbW} height={mbKbH} fill="rgba(0,0,0,0.2)" rx="2" />
      {/* Trackpad */}
      <rect x={mbTpX} y={mbTpY} width={mbTpW} height={mbTpH} fill="rgba(255,255,255,0.08)" rx="2" />
    </g>
  );
};
