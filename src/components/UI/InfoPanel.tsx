import React, { useState } from "react";

interface InfoPanelProps {
  title: string;
  content: React.ReactNode;
  position?: "top-right" | "top-left" | "bottom-right" | "bottom-left" | "top" | "bottom";
  size?: "small" | "medium" | "large";
}

export default function InfoPanel({ 
  title, 
  content, 
  position = "top-right",
  size = "medium"
}: InfoPanelProps) {
  const [isOpen, setIsOpen] = useState(false);

  const sizeStyles = {
    small: { width: "280px", maxHeight: "200px" },
    medium: { width: "350px", maxHeight: "400px" },
    large: { width: "450px", maxHeight: "500px" }
  };

  const positionStyles: Record<string, React.CSSProperties> = {
    "top-right": { top: "10px", right: "10px" },
    "top-left": { top: "10px", left: "10px" },
    "bottom-right": { bottom: "10px", right: "10px" },
    "bottom-left": { bottom: "10px", left: "10px" },
    "top": { top: "10px", left: "50%", transform: "translateX(-50%)" },
    "bottom": { bottom: "10px", left: "50%", transform: "translateX(-50%)" }
  };

  return (
    <>
      {/* Info icon button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        style={{
          position: "relative",
          width: "24px",
          height: "24px",
          borderRadius: "50%",
          border: "2px solid #4a90e2",
          background: isOpen ? "#4a90e2" : "white",
          color: isOpen ? "white" : "#4a90e2",
          fontSize: "16px",
          fontWeight: "bold",
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          transition: "all 0.2s",
          zIndex: 1000
        }}
        title={`${isOpen ? "Hide" : "Show"} information about ${title}`}
        aria-label={`${isOpen ? "Hide" : "Show"} info panel`}
      >
        ?
      </button>

      {/* Info panel */}
      {isOpen && (
        <>
          {/* Backdrop */}
          <div
            style={{
              position: "fixed",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: "rgba(0, 0, 0, 0.3)",
              zIndex: 9998
            }}
            onClick={() => setIsOpen(false)}
          />

          {/* Panel content */}
          <div
            style={{
              position: "fixed",
              ...positionStyles[position],
              ...sizeStyles[size],
              background: "white",
              border: "2px solid #4a90e2",
              borderRadius: "8px",
              padding: "20px",
              boxShadow: "0 4px 20px rgba(0,0,0,0.2)",
              zIndex: 9999,
              overflowY: "auto",
              maxWidth: "90vw"
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "12px" }}>
              <h3 style={{ margin: 0, color: "#4a90e2", fontSize: "18px" }}>{title}</h3>
              <button
                onClick={() => setIsOpen(false)}
                style={{
                  background: "none",
                  border: "none",
                  fontSize: "24px",
                  cursor: "pointer",
                  color: "#666",
                  padding: "0",
                  marginLeft: "12px",
                  lineHeight: "1"
                }}
                aria-label="Close info panel"
              >
                ×
              </button>
            </div>
            
            <div style={{ fontSize: "14px", lineHeight: "1.6", color: "#333" }}>
              {content}
            </div>
          </div>
        </>
      )}
    </>
  );
}

