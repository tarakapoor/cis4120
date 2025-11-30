import React, { useState } from "react";
import { createPortal } from "react-dom";
import { getDefinition } from "../../data/glossary";

interface TermDefinitionProps {
  term: string;
  children?: React.ReactNode;
  style?: React.CSSProperties;
}

export default function TermDefinition({ term, children, style }: TermDefinitionProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [position, setPosition] = useState<{ x: number; y: number } | null>(null);
  const definition = getDefinition(term);

  if (!definition) {
    // If no definition found, just return the term or children as-is
    return <span style={style}>{children || term}</span>;
  }

  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    setPosition({ x: rect.left, y: rect.top + rect.height + 5 });
    setIsOpen(!isOpen);
  };

  const handleClose = () => {
    setIsOpen(false);
  };

  return (
    <>
      <span
        onClick={handleClick}
        style={{
          color: "#2563eb",
          textDecoration: "underline",
          textDecorationStyle: "dashed",
          cursor: "help",
          ...style
        }}
        title={`Click to learn about "${term}"`}
      >
        {children || term}
      </span>

      {isOpen && position && typeof document !== 'undefined' && createPortal(
        <>
          {/* Backdrop to close on click outside */}
          <div
            style={{
              position: "fixed",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              zIndex: 9998,
              cursor: "default"
            }}
            onClick={handleClose}
          />
          
          {/* Definition popup */}
          <div
            style={{
              position: "fixed",
              left: `${Math.min(position.x, window.innerWidth - 350)}px`,
              top: `${Math.min(position.y, window.innerHeight - 200)}px`,
              maxWidth: "320px",
              background: "white",
              border: "2px solid #2563eb",
              borderRadius: "8px",
              padding: "16px",
              boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
              zIndex: 9999,
              fontSize: "14px",
              lineHeight: "1.6"
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "8px" }}>
              <strong style={{ color: "#2563eb", fontSize: "16px", fontWeight: "600" }}>{definition.term}</strong>
              <button
                onClick={handleClose}
                style={{
                  background: "none",
                  border: "none",
                  fontSize: "20px",
                  cursor: "pointer",
                  color: "#666",
                  padding: "0",
                  marginLeft: "8px",
                  lineHeight: "1"
                }}
                aria-label="Close definition"
              >
                ×
              </button>
            </div>
            
            <div style={{ margin: "8px 0", color: "#475569" }}>{definition.definition}</div>
            
            {definition.example && (
              <div style={{ marginTop: "12px", padding: "8px", background: "#f8fafc", borderRadius: "4px", border: "1px solid #e2e8f0" }}>
                <strong style={{ fontSize: "12px", color: "#64748b" }}>Example:</strong>
                <div style={{ margin: "4px 0 0 0", fontSize: "13px", color: "#475569", fontStyle: "italic" }}>
                  {definition.example}
                </div>
              </div>
            )}

            {definition.relatedTerms && definition.relatedTerms.length > 0 && (
              <div style={{ marginTop: "12px", fontSize: "12px" }}>
                <strong style={{ color: "#64748b" }}>Related terms: </strong>
                <span style={{ color: "#94a3b8" }}>
                  {definition.relatedTerms.join(", ")}
                </span>
              </div>
            )}
          </div>
        </>,
        document.body
      )}
    </>
  );
}

