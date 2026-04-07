import { useEffect, useRef } from 'react';
import styled from 'styled-components';

export interface NetworkMeshProps {
  /** Number of particles. Default 60. */
  count?: number;
  /** Max distance for drawing connections. Default 150. */
  linkDistance?: number;
  /** Particle color. Default 'rgba(255,215,0,0.15)'. */
  color?: string;
  /** Connection color. Default 'rgba(255,215,0,0.04)'. */
  lineColor?: string;
  /** Particle radius. Default 1.5. */
  radius?: number;
  /** Max drift speed in px/frame. Default 0.3. */
  speed?: number;
  className?: string;
}

const Canvas = styled.canvas`
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
`;

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
}

export function NetworkMesh({
  count = 60,
  linkDistance = 150,
  color = 'rgba(255,215,0,0.15)',
  lineColor = 'rgba(255,215,0,0.04)',
  radius = 1.5,
  speed = 0.3,
  className,
}: NetworkMeshProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let w = 0;
    let h = 0;

    function resize() {
      w = window.innerWidth;
      h = window.innerHeight;
      canvas!.width = w * devicePixelRatio;
      canvas!.height = h * devicePixelRatio;
      canvas!.style.width = `${w}px`;
      canvas!.style.height = `${h}px`;
      ctx!.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);

      // Re-scatter particles that fell outside bounds
      for (const p of particlesRef.current) {
        if (p.x > w) p.x = Math.random() * w;
        if (p.y > h) p.y = Math.random() * h;
      }
    }

    function init() {
      resize();
      const particles: Particle[] = [];
      for (let i = 0; i < count; i++) {
        particles.push({
          x: Math.random() * w,
          y: Math.random() * h,
          vx: (Math.random() - 0.5) * speed * 2,
          vy: (Math.random() - 0.5) * speed * 2,
        });
      }
      particlesRef.current = particles;
    }

    function tick() {
      const particles = particlesRef.current;
      ctx!.clearRect(0, 0, w, h);

      // Update positions
      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;

        // Soft bounce off edges
        if (p.x < 0 || p.x > w) p.vx *= -1;
        if (p.y < 0 || p.y > h) p.vy *= -1;

        p.x = Math.max(0, Math.min(w, p.x));
        p.y = Math.max(0, Math.min(h, p.y));
      }

      // Draw connections. Use globalAlpha for the distance-fade so we don't need
      // to parse lineColor — any valid CSS color (hex, rgb, rgba, named) works.
      const linkDist2 = linkDistance * linkDistance;
      ctx!.strokeStyle = lineColor;

      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const d2 = dx * dx + dy * dy;
          if (d2 < linkDist2) {
            const fade = 1 - Math.sqrt(d2) / linkDistance;
            ctx!.globalAlpha = fade;
            ctx!.beginPath();
            ctx!.moveTo(particles[i].x, particles[i].y);
            ctx!.lineTo(particles[j].x, particles[j].y);
            ctx!.stroke();
          }
        }
      }
      ctx!.globalAlpha = 1;

      // Draw particles
      ctx!.fillStyle = color;
      for (const p of particles) {
        ctx!.beginPath();
        ctx!.arc(p.x, p.y, radius, 0, Math.PI * 2);
        ctx!.fill();
      }

      rafRef.current = requestAnimationFrame(tick);
    }

    init();
    rafRef.current = requestAnimationFrame(tick);
    window.addEventListener('resize', resize);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener('resize', resize);
    };
  }, [count, linkDistance, color, lineColor, radius, speed]);

  return <Canvas ref={canvasRef} className={className} />;
}
