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

export default function NetworkGraph({ model }: any) {
  const ref = useRef(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);

  useEffect(() => {
    if (!model || !model.layers) return;

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

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
        const [x, y] = d3.pointer(event);
        setTooltip({ x, y, text: d.id });
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
    <div style={{ position: "relative", width: 900, height: 600 }}>
      {tooltip && (
        <div
          style={{
            position: "absolute",
            top: tooltip.y + 10,
            left: tooltip.x + 10,
            background: "white",
            padding: "4px 8px",
            borderRadius: "4px",
            border: "1px solid #ccc",
            fontSize: "12px",
            pointerEvents: "none",
            zIndex: 10
          }}
        >
          {tooltip.text}
        </div>
      )}

      <svg ref={ref} width={900} height={600} />
    </div>
  );
}