import React, { useEffect, useRef } from "react";

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

    // Animation variables
    let animationFrame: number;
    let time = 0;

    // Gradient mesh points
    const points: Array<{ x: number; y: number; vx: number; vy: number }> = [];
    const numPoints = 8;

    // Initialize points with better distribution
    for (let i = 0; i < numPoints; i++) {
      points.push({
        x: (Math.random() * 0.6 + 0.2) * canvas.width, // Keep away from edges
        y: (Math.random() * 0.6 + 0.2) * canvas.height,
        vx: (Math.random() - 0.5) * 0.8,
        vy: (Math.random() - 0.5) * 0.8
      });
    }

    const draw = () => {
      time += 0.008;
      
      // Clear with subtle base color
      ctx.fillStyle = "#f8fafc";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Update point positions with slower, smoother movement
      points.forEach((point, i) => {
        // Add some sine wave variation for smoother movement
        point.x += point.vx + Math.sin(time + i) * 0.3;
        point.y += point.vy + Math.cos(time + i) * 0.3;

        // Bounce off edges
        if (point.x < 0 || point.x > canvas.width) point.vx *= -1;
        if (point.y < 0 || point.y > canvas.height) point.vy *= -1;

        // Keep in bounds
        point.x = Math.max(0, Math.min(canvas.width, point.x));
        point.y = Math.max(0, Math.min(canvas.height, point.y));
      });

      // Draw more visible gradient mesh
      const gradient1 = ctx.createRadialGradient(
        points[0].x,
        points[0].y,
        0,
        points[0].x,
        points[0].y,
        Math.max(canvas.width, canvas.height) * 0.5
      );
      gradient1.addColorStop(0, "rgba(37, 99, 235, 0.15)");
      gradient1.addColorStop(0.4, "rgba(147, 197, 253, 0.08)");
      gradient1.addColorStop(0.8, "rgba(203, 213, 225, 0.03)");
      gradient1.addColorStop(1, "rgba(248, 250, 252, 0)");

      const gradient2 = ctx.createRadialGradient(
        points[1].x,
        points[1].y,
        0,
        points[1].x,
        points[1].y,
        Math.max(canvas.width, canvas.height) * 0.45
      );
      gradient2.addColorStop(0, "rgba(30, 64, 175, 0.12)");
      gradient2.addColorStop(0.4, "rgba(100, 116, 139, 0.06)");
      gradient2.addColorStop(0.8, "rgba(148, 163, 184, 0.02)");
      gradient2.addColorStop(1, "rgba(248, 250, 252, 0)");

      const gradient3 = ctx.createRadialGradient(
        points[2].x,
        points[2].y,
        0,
        points[2].x,
        points[2].y,
        Math.max(canvas.width, canvas.height) * 0.5
      );
      gradient3.addColorStop(0, "rgba(59, 130, 246, 0.13)");
      gradient3.addColorStop(0.4, "rgba(191, 219, 254, 0.06)");
      gradient3.addColorStop(0.8, "rgba(226, 232, 240, 0.02)");
      gradient3.addColorStop(1, "rgba(248, 250, 252, 0)");

      // Additional gradients for more depth
      const gradient4 = ctx.createRadialGradient(
        points[3]?.x || canvas.width * 0.3,
        points[3]?.y || canvas.height * 0.7,
        0,
        points[3]?.x || canvas.width * 0.3,
        points[3]?.y || canvas.height * 0.7,
        Math.max(canvas.width, canvas.height) * 0.4
      );
      gradient4.addColorStop(0, "rgba(71, 85, 105, 0.08)");
      gradient4.addColorStop(0.5, "rgba(148, 163, 184, 0.04)");
      gradient4.addColorStop(1, "rgba(248, 250, 252, 0)");

      // Draw gradients with blend mode for smoother effect
      ctx.globalCompositeOperation = "screen";
      ctx.fillStyle = gradient1;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.fillStyle = gradient2;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.fillStyle = gradient3;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.fillStyle = gradient4;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.globalCompositeOperation = "source-over";

      // Draw subtle connecting lines between points
      ctx.lineWidth = 1.5;
      
      for (let i = 0; i < points.length; i++) {
        for (let j = i + 1; j < points.length; j++) {
          const dx = points[i].x - points[j].x;
          const dy = points[i].y - points[j].y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          // Only draw lines for nearby points
          if (distance < Math.max(canvas.width, canvas.height) * 0.35) {
            const opacity = (1 - distance / (Math.max(canvas.width, canvas.height) * 0.35)) * 0.08;
            ctx.strokeStyle = `rgba(37, 99, 235, ${opacity})`;
            ctx.beginPath();
            ctx.moveTo(points[i].x, points[i].y);
            ctx.lineTo(points[j].x, points[j].y);
            ctx.stroke();
          }
        }
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

