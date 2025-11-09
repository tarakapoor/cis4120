// import React, { useEffect, useRef } from "react";
// import * as d3 from "d3";

// export default function NetworkGraph({ model }: any) {
//   const ref = useRef(null);

//   useEffect(() => {
//     if (!model || !model.layers) return;

//     const svg = d3.select(ref.current);
//     svg.selectAll("*").remove();

//     const width = 900;
//     const height = 600;

//     const layers = model.layers;
//     const layerSpacing = width / (layers.length + 1);

//     const nodes: any[] = [];
//     const links: any[] = [];

//     layers.forEach((layer: any, layerIdx: number) => {
//       const ySpacing = height / (layer.size + 1);
//       for (let i = 0; i < layer.size; i++) {
//         nodes.push({
//           id: `L${layerIdx}-N${i}`,
//           layer: layerIdx,
//           x: (layerIdx + 1) * layerSpacing,
//           y: (i + 1) * ySpacing
//         });
//       }
//     });

//     nodes.forEach((n) => {
//       const layer = n.layer;
//       if (layer < layers.length - 1) {
//         nodes
//           .filter((m) => m.layer === layer + 1)
//           .forEach((next) => {
//             links.push({ source: n, target: next });
//           });
//       }
//     });

//     const g = svg.append("g");

//     const zoom = d3.zoom().on("zoom", (event) => {
//       g.attr("transform", event.transform);
//     });
//     svg.call(zoom as any);

//     g.selectAll("line")
//       .data(links)
//       .enter()
//       .append("line")
//       .attr("x1", (d: any) => d.source.x)
//       .attr("y1", (d: any) => d.source.y)
//       .attr("x2", (d: any) => d.target.x)
//       .attr("y2", (d: any) => d.target.y)
//       .attr("stroke", "#aaa");

//     g.selectAll("circle")
//       .data(nodes)
//       .enter()
//       .append("circle")
//       .attr("cx", (d: any) => d.x)
//       .attr("cy", (d: any) => d.y)
//       .attr("r", 10)
//       .attr("fill", "#4a90e2");

//   }, [model]);

//   return <svg ref={ref} width={900} height={600} />;
// }

import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import InfoPanel from "../UI/InfoPanel";
import TermDefinition from "../UI/TermDefinition";

export default function NetworkGraph({ model }: any) {
  const ref = useRef(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);

  useEffect(() => {
    if (!model || !model.layers) return;

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    // Get SVG dimensions from its container
    const svgElement = ref.current as SVGSVGElement;
    if (!svgElement) return;
    
    // Use fixed dimensions for calculation, SVG will scale via CSS
    const width = 900;
    const height = 600;

    const layers = model.layers;
    const layerSpacing = width / (layers.length + 1);

    const nodes: any[] = [];
    const links: any[] = [];

    // Build nodes
    layers.forEach((layer: any, layerIdx: number) => {
      const ySpacing = height / (layer.size + 1);
      for (let i = 0; i < layer.size; i++) {
        nodes.push({
          id: `L${layerIdx}-N${i}`,
          layer: layerIdx,
          x: (layerIdx + 1) * layerSpacing,
          y: (i + 1) * ySpacing
        });
      }
    });

    // Build edges
    nodes.forEach((n) => {
      const layer = n.layer;
      if (layer < layers.length - 1) {
        nodes
          .filter((m) => m.layer === layer + 1)
          .forEach((next) => {
            links.push({ source: n, target: next });
          });
      }
    });

    const g = svg.append("g");

    // Zoom/pan
    const zoom = d3.zoom().on("zoom", (event) => {
      g.attr("transform", event.transform);
    });
    svg.call(zoom as any);

    // --- DRAW LINKS ---
    const linkElems = g
      .selectAll("line")
      .data(links)
      .enter()
      .append("line")
      .attr("x1", (d: any) => d.source.x)
      .attr("y1", (d: any) => d.source.y)
      .attr("x2", (d: any) => d.target.x)
      .attr("y2", (d: any) => d.target.y)
      .attr("stroke", "#aaa")
      .attr("stroke-width", 1.2)
      .attr("opacity", 0.7);

    // --- DRAW NODES ---
    const nodeElems = g
      .selectAll("circle")
      .data(nodes)
      .enter()
      .append("circle")
      .attr("cx", (d: any) => d.x)
      .attr("cy", (d: any) => d.y)
      .attr("r", 11)
      .attr("fill", "#4a90e2")
      .style("cursor", "pointer")

      // Tooltip
      .on("mouseenter", (event, d) => {
        const svgElement = ref.current as SVGSVGElement;
        if (!svgElement) return;
        const rect = svgElement.getBoundingClientRect();
        const [x, y] = d3.pointer(event);
        setTooltip({ x: x + rect.left, y: y + rect.top, text: d.id });
      })
      .on("mouseleave", () => setTooltip(null))

      // Click node → highlight
      .on("click", (_, d) => {
        setSelected((prev) => (prev === d.id ? null : d.id));
      });

    // --- HANDLE HIGHLIGHT LOGIC ---
    const updateHighlight = () => {
      if (!selected) {
        // reset everything
        linkElems
          .attr("stroke", "#aaa")
          .attr("stroke-width", 1.2)
          .attr("opacity", 0.7);

        nodeElems
          .attr("fill", "#4a90e2")
          .attr("opacity", 1);
        return;
      }

      // Find connected edges
      const outgoing = links.filter((l) => l.source.id === selected);
      const incoming = links.filter((l) => l.target.id === selected);

      const connectedNodeIds = new Set([
        ...outgoing.map((l) => l.target.id),
        ...incoming.map((l) => l.source.id),
        selected
      ]);

      // Highlight nodes
      nodeElems
        .attr("fill", (d: any) =>
          d.id === selected
            ? "#ff4136" // selected = red
            : connectedNodeIds.has(d.id)
            ? "#ff9f43" // connected = orange
            : "#4a90e2" // others = blue
        )
        .attr("opacity", (d: any) => (connectedNodeIds.has(d.id) ? 1 : 0.25));

      // Highlight edges
      linkElems
        .attr("stroke", (d: any) =>
          d.source.id === selected || d.target.id === selected ? "#ff4136" : "#aaa"
        )
        .attr("stroke-width", (d: any) =>
          d.source.id === selected || d.target.id === selected ? 3 : 1
        )
        .attr("opacity", (d: any) =>
          d.source.id === selected || d.target.id === selected ? 1 : 0.1
        );
    };

    updateHighlight();
  }, [model, selected]);

  return (
    <div style={{ position: "relative", width: "100%", height: "100%", padding: "20px" }}>
      <div style={{ marginBottom: "16px", display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <h2 style={{ margin: "0 0 8px 0", fontSize: "24px" }}>
            <TermDefinition term="neural network">Neural Network</TermDefinition> Visualization
          </h2>
          <p style={{ margin: 0, color: "#666", fontSize: "14px" }}>
            Explore your network by clicking on <TermDefinition term="neuron">neurons</TermDefinition> to see their <TermDefinition term="connection">connections</TermDefinition>
          </p>
        </div>
        <InfoPanel
          title="How to Use the Network Visualization"
          content={
            <div>
              <p style={{ marginTop: 0 }}>
                <strong>Understanding the Visualization:</strong>
              </p>
              <ul style={{ paddingLeft: "20px", margin: "8px 0" }}>
                <li><strong>Circles</strong> represent <TermDefinition term="neuron">neurons</TermDefinition> (nodes) in the network</li>
                <li><strong>Lines</strong> represent <TermDefinition term="edge">edges</TermDefinition> (connections) between neurons</li>
                <li><strong>Blue nodes</strong> are normal neurons</li>
                <li><strong>Red nodes</strong> are selected neurons</li>
                <li><strong>Orange nodes</strong> are connected to the selected neuron</li>
              </ul>

              <p>
                <strong>Interactions:</strong>
              </p>
              <ul style={{ paddingLeft: "20px", margin: "8px 0" }}>
                <li><strong>Click a neuron</strong> to highlight it and see all its connections</li>
                <li><strong>Click again</strong> to deselect and see the full network</li>
                <li><strong>Hover over neurons</strong> to see their IDs</li>
                <li><strong>Scroll to zoom</strong> in and out of the network</li>
                <li><strong>Click and drag</strong> to pan around the visualization</li>
              </ul>

              <div style={{ marginTop: "16px", padding: "12px", background: "#e8f4f8", borderRadius: "4px" }}>
                <strong>💡 Interpreting Activation Values:</strong>
                <p style={{ margin: "8px 0 0 0", fontSize: "13px" }}>
                  <TermDefinition term="activation">Activation</TermDefinition> values tell you how strongly a <TermDefinition term="neuron">neuron</TermDefinition> responds to input. 
                  Higher values (closer to 1) indicate stronger responses, while lower values (closer to 0) indicate weaker responses. 
                  When you select a neuron, you're seeing how it connects to other parts of the network.
                </p>
              </div>

              <div style={{ marginTop: "12px", padding: "12px", background: "#fff3cd", borderRadius: "4px" }}>
                <strong>🎯 Best Practices:</strong>
                <ul style={{ margin: "8px 0 0 0", paddingLeft: "20px", fontSize: "13px" }}>
                  <li>Start by exploring the <TermDefinition term="input">input layer</TermDefinition> to understand data flow</li>
                  <li>Follow connections through <TermDefinition term="hidden layer">hidden layers</TermDefinition> to see how information transforms</li>
                  <li>Check the <TermDefinition term="output">output layer</TermDefinition> to see final predictions</li>
                  <li>Look for patterns in how neurons connect - dense connections indicate important relationships</li>
                </ul>
              </div>

              <div style={{ marginTop: "12px", padding: "12px", background: "#d4edda", borderRadius: "4px" }}>
                <strong>📚 Key Terms:</strong>
                <p style={{ margin: "8px 0 0 0", fontSize: "13px" }}>
                  Click on any underlined term (like <TermDefinition term="activation">activation</TermDefinition> or <TermDefinition term="layer">layer</TermDefinition>) 
                  throughout the interface to learn what it means in simple terms.
                </p>
              </div>
            </div>
          }
          position="top-right"
          size="large"
        />
      </div>

      <div style={{ 
        border: "1px solid #ddd", 
        borderRadius: "8px", 
        background: "white",
        position: "relative",
        width: "100%",
        height: "calc(100% - 120px)",
        minHeight: "600px"
      }}>
        {tooltip && (
          <div
            style={{
              position: "fixed",
              top: tooltip.y + 10,
              left: tooltip.x + 10,
              background: "white",
              padding: "6px 10px",
              borderRadius: "4px",
              border: "1px solid #4a90e2",
              fontSize: "12px",
              pointerEvents: "none",
              zIndex: 10,
              boxShadow: "0 2px 8px rgba(0,0,0,0.1)"
            }}
          >
            <strong>Neuron:</strong> {tooltip.text}
          </div>
        )}

        <svg 
          ref={ref} 
          viewBox="0 0 900 600" 
          preserveAspectRatio="xMidYMid meet"
          style={{ display: "block", width: "100%", height: "100%" }} 
        />
      </div>

      {selected && (
        <div style={{
          marginTop: "16px",
          padding: "12px",
          background: "#f0f7ff",
          border: "1px solid #4a90e2",
          borderRadius: "4px",
          fontSize: "14px"
        }}>
          <strong>Selected:</strong> {selected}
          <p style={{ margin: "8px 0 0 0", fontSize: "13px", color: "#666" }}>
            This <TermDefinition term="neuron">neuron</TermDefinition> is connected to other neurons through <TermDefinition term="edge">edges</TermDefinition>. 
            The highlighted connections show how information flows from this neuron to others in the network.
          </p>
        </div>
      )}
    </div>
  );
}