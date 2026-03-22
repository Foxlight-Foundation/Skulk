/**
 * FamilyLogos
 *
 * SVG icon for each model family.
 * Ported from FamilyLogos.svelte.
 */
import React from 'react';

export interface FamilyLogoProps {
  family: string;
  size?: number;
  className?: string;
}

export const FamilyLogo: React.FC<FamilyLogoProps> = ({ family, size = 20, className }) => {
  const f = family.toLowerCase();

  if (f === 'favorites') {
    return (
      <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
        <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"
          fill="oklch(0.85 0.18 85)" />
      </svg>
    );
  }

  if (f === 'recents') {
    return (
      <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
        <circle cx="12" cy="12" r="10" stroke="oklch(0.6 0 0)" strokeWidth="1.5" />
        <path d="M12 6v6l4 2" stroke="oklch(0.85 0.18 85)" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    );
  }

  if (f === 'llama' || f === 'meta') {
    return (
      <svg width={size} height={size} viewBox="0 0 40 40" fill="none" className={className}>
        <circle cx="20" cy="20" r="18" fill="#0866ff" />
        <text x="20" y="26" textAnchor="middle" fontSize="16" fill="white" fontWeight="bold">M</text>
      </svg>
    );
  }

  if (f === 'qwen') {
    return (
      <svg width={size} height={size} viewBox="0 0 40 40" fill="none" className={className}>
        <circle cx="20" cy="20" r="18" fill="#6e40c9" />
        <text x="20" y="26" textAnchor="middle" fontSize="14" fill="white" fontWeight="bold">Q</text>
      </svg>
    );
  }

  if (f === 'deepseek') {
    return (
      <svg width={size} height={size} viewBox="0 0 40 40" fill="none" className={className}>
        <circle cx="20" cy="20" r="18" fill="#1d4ed8" />
        <text x="20" y="26" textAnchor="middle" fontSize="13" fill="white" fontWeight="bold">DS</text>
      </svg>
    );
  }

  if (f === 'mistral') {
    return (
      <svg width={size} height={size} viewBox="0 0 40 40" fill="none" className={className}>
        <circle cx="20" cy="20" r="18" fill="#f97316" />
        <text x="20" y="26" textAnchor="middle" fontSize="13" fill="white" fontWeight="bold">Mi</text>
      </svg>
    );
  }

  if (f === 'flux' || f === 'stable diffusion' || f === 'sd') {
    return (
      <svg width={size} height={size} viewBox="0 0 40 40" fill="none" className={className}>
        <circle cx="20" cy="20" r="18" fill="#7c3aed" />
        <text x="20" y="26" textAnchor="middle" fontSize="13" fill="white" fontWeight="bold">Fx</text>
      </svg>
    );
  }

  if (f === 'huggingface' || f === 'hf') {
    return (
      <svg width={size} height={size} viewBox="0 0 40 40" fill="none" className={className}>
        <circle cx="20" cy="20" r="18" fill="#ffd21e" />
        <text x="20" y="26" textAnchor="middle" fontSize="18" fill="#333">🤗</text>
      </svg>
    );
  }

  // Generic fallback
  const initials = family.slice(0, 2).toUpperCase();
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" fill="none" className={className}>
      <circle cx="20" cy="20" r="18" fill="oklch(0.25 0 0)" stroke="oklch(0.35 0 0)" strokeWidth="1" />
      <text x="20" y="26" textAnchor="middle" fontSize="13" fill="oklch(0.7 0 0)" fontWeight="bold">
        {initials}
      </text>
    </svg>
  );
};
