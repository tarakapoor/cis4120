import React, { useEffect, useRef } from "react";

interface Node {
  x: number;
  y: number;
  vx: number;
  vy: number;
  pulsePhase: number;
  pulseSpeed: number;
  connections: number[];
}

export default function DynamicBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    // Color scheme: darker, cooler toned blues
    const colors = {
      primary: "#0f172a",      // Very dark navy
      secondary: "#1e293b",     // Dark navy
      accent: "#475569",       // Medium slate blue
      glow: "#64748b"          // Light slate for glow
    };

    // Animation variables
    let animationFrame: number;
    let time = 0;
    const numNodes = 25;
    const nodes: Node[] = [];

    // Initialize nodes
    for (let i = 0; i < numNodes; i++) {
      nodes.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        pulsePhase: Math.random() * Math.PI * 2,
        pulseSpeed: 0.02 + Math.random() * 0.03,
        connections: []
      });
    }

    // Build connection network (each node connects to nearby nodes)
    const buildConnections = () => {
      nodes.forEach((node, i) => {
        node.connections = [];
        nodes.forEach((other, j) => {
          if (i !== j) {
            const dx = node.x - other.x;
            const dy = node.y - other.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            const maxDist = Math.min(canvas.width, canvas.height) * 0.4;
            if (dist < maxDist && Math.random() > 0.7) {
              node.connections.push(j);
            }
          }
        });
      });
    };
    buildConnections();

    const draw = () => {
      time += 0.01;
      
      // Clear with very dark navy base
      ctx.fillStyle = "#0c1226";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Update node positions
      nodes.forEach((node) => {
        node.x += node.vx;
        node.y += node.vy;
        node.pulsePhase += node.pulseSpeed;

        // Bounce off edges
        if (node.x < 0 || node.x > canvas.width) {
          node.vx *= -1;
          node.x = Math.max(0, Math.min(canvas.width, node.x));
        }
        if (node.y < 0 || node.y > canvas.height) {
          node.vy *= -1;
          node.y = Math.max(0, Math.min(canvas.height, node.y));
        }

        // Keep in bounds
        node.x = Math.max(0, Math.min(canvas.width, node.x));
        node.y = Math.max(0, Math.min(canvas.height, node.y));
      });

      // Draw connections first (so nodes appear on top)
      ctx.lineWidth = 1.5;
      nodes.forEach((node, i) => {
        node.connections.forEach((targetIdx) => {
          const target = nodes[targetIdx];
          const dx = node.x - target.x;
          const dy = node.y - target.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          const maxDist = Math.min(canvas.width, canvas.height) * 0.4;
          
          if (dist < maxDist) {
            // Calculate opacity based on distance and pulse
            const baseOpacity = 1 - (dist / maxDist);
            const pulse = (Math.sin(time * 2 + node.pulsePhase) + 1) * 0.5;
            const opacity = baseOpacity * (0.3 + pulse * 0.2);
            
            // Create gradient for the connection with cooler blues
            const gradient = ctx.createLinearGradient(node.x, node.y, target.x, target.y);
            gradient.addColorStop(0, `rgba(71, 85, 105, ${opacity})`);
            gradient.addColorStop(0.5, `rgba(100, 116, 139, ${opacity * 0.8})`);
            gradient.addColorStop(1, `rgba(71, 85, 105, ${opacity})`);
            
            ctx.strokeStyle = gradient;
            ctx.beginPath();
            ctx.moveTo(node.x, node.y);
            ctx.lineTo(target.x, target.y);
            ctx.stroke();

            // Draw flowing particles along connections
            const numParticles = 2;
            for (let p = 0; p < numParticles; p++) {
              const particlePos = ((time * 0.3 + p * 0.5) % 1);
              const px = node.x + (target.x - node.x) * particlePos;
              const py = node.y + (target.y - node.y) * particlePos;
              
              ctx.fillStyle = `rgba(148, 163, 184, ${opacity * 0.8})`;
              ctx.beginPath();
              ctx.arc(px, py, 2, 0, Math.PI * 2);
              ctx.fill();
            }
          }
        });
      });

      // Draw nodes with pulsing glow
      nodes.forEach((node) => {
        const pulse = (Math.sin(node.pulsePhase) + 1) * 0.5;
        const nodeSize = 4 + pulse * 3;
        const glowSize = nodeSize + 8 + pulse * 6;
        
        // Outer glow
        const glowGradient = ctx.createRadialGradient(
          node.x, node.y, 0,
          node.x, node.y, glowSize
        );
        glowGradient.addColorStop(0, `rgba(148, 163, 184, ${0.4 + pulse * 0.3})`);
        glowGradient.addColorStop(0.5, `rgba(100, 116, 139, ${0.2 + pulse * 0.2})`);
        glowGradient.addColorStop(1, "rgba(71, 85, 105, 0)");
        
        ctx.fillStyle = glowGradient;
        ctx.beginPath();
        ctx.arc(node.x, node.y, glowSize, 0, Math.PI * 2);
        ctx.fill();

        // Node core
        const coreGradient = ctx.createRadialGradient(
          node.x, node.y, 0,
          node.x, node.y, nodeSize
        );
        coreGradient.addColorStop(0, colors.accent);
        coreGradient.addColorStop(0.5, colors.secondary);
        coreGradient.addColorStop(1, colors.primary);
        
        ctx.fillStyle = coreGradient;
        ctx.beginPath();
        ctx.arc(node.x, node.y, nodeSize, 0, Math.PI * 2);
        ctx.fill();

        // Bright center
        ctx.fillStyle = colors.glow;
        ctx.beginPath();
        ctx.arc(node.x, node.y, nodeSize * 0.4, 0, Math.PI * 2);
        ctx.fill();
      });

      // Occasionally rebuild connections for dynamic network
      if (Math.floor(time * 10) % 300 === 0) {
        buildConnections();
      }

      animationFrame = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      window.removeEventListener("resize", resizeCanvas);
      cancelAnimationFrame(animationFrame);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100vw",
        height: "100vh",
        zIndex: 0,
        pointerEvents: "none",
        display: "block"
      }}
    />
  );
}
