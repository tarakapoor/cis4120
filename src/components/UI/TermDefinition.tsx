import React, { useState } from "react";
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
          color: "#4a90e2",
          textDecoration: "underline",
          textDecorationStyle: "dashed",
          cursor: "help",
          ...style
        }}
        title={`Click to learn about "${term}"`}
      >
        {children || term}
      </span>

      {isOpen && position && (
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
              border: "2px solid #4a90e2",
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
              <strong style={{ color: "#4a90e2", fontSize: "16px" }}>{definition.term}</strong>
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
            
            <p style={{ margin: "8px 0", color: "#333" }}>{definition.definition}</p>
            
            {definition.example && (
              <div style={{ marginTop: "12px", padding: "8px", background: "#f5f5f5", borderRadius: "4px" }}>
                <strong style={{ fontSize: "12px", color: "#666" }}>Example:</strong>
                <p style={{ margin: "4px 0 0 0", fontSize: "13px", color: "#555", fontStyle: "italic" }}>
                  {definition.example}
                </p>
              </div>
            )}

            {definition.relatedTerms && definition.relatedTerms.length > 0 && (
              <div style={{ marginTop: "12px", fontSize: "12px" }}>
                <strong style={{ color: "#666" }}>Related terms: </strong>
                <span style={{ color: "#888" }}>
                  {definition.relatedTerms.join(", ")}
                </span>
              </div>
            )}
          </div>
        </>
      )}
    </>
  );
}

