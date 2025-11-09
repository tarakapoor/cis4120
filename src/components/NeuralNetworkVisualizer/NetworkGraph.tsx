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
import WeightAdjustmentPanel from "./WeightAdjustmentPanel";
import { 
  initializeWeights, 
  getWeight, 
  getWeightColor, 
  getWeightStrokeWidth,
  WeightData,
  ModelWithWeights
} from "../../utils/modelUtils";

export default function NetworkGraph({ model }: any) {
  const ref = useRef(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);
  
  // Initialize model with weights and store in state
  const [modelWithWeights, setModelWithWeights] = useState<ModelWithWeights | null>(null);
  const [originalWeights, setOriginalWeights] = useState<WeightData[]>([]);

  // Initialize weights when model changes
  useEffect(() => {
    if (model) {
      const initialized = initializeWeights(model);
      setModelWithWeights(initialized);
      // Store original weights for comparison
      setOriginalWeights(initialized.weights ? [...initialized.weights] : []);
    }
  }, [model]);

  const handleWeightChange = (newWeights: WeightData[]) => {
    if (modelWithWeights) {
      setModelWithWeights({
        ...modelWithWeights,
        weights: newWeights
      });
    }
  };

  useEffect(() => {
    if (!modelWithWeights || !modelWithWeights.layers) return;

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    // Get SVG dimensions from its container
    const svgElement = ref.current as SVGSVGElement;
    if (!svgElement) return;
    
    // Use fixed dimensions for calculation, SVG will scale via CSS
    const width = 900;
    const height = 600;

    const layers = modelWithWeights.layers;
    const weights = modelWithWeights.weights || [];
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

    // Build edges with weights
    nodes.forEach((n) => {
      const layer = n.layer;
      if (layer < layers.length - 1) {
        nodes
          .filter((m) => m.layer === layer + 1)
          .forEach((next) => {
            const weight = getWeight(weights, n.id, next.id);
            links.push({ 
              source: n, 
              target: next,
              weight: weight,
              id: `${n.id}-${next.id}`
            });
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
      .attr("stroke", (d: any) => {
        // Use weight-based coloring
        if (selected && (d.source.id === selected || d.target.id === selected)) {
          return getWeightColor(d.weight);
        }
        // For non-selected edges, use weight-based color but extract RGB and set lower opacity
        if (d.weight > 0) {
          return "rgba(74, 144, 226, 0.3)";
        } else if (d.weight < 0) {
          return "rgba(255, 65, 54, 0.3)";
        } else {
          return "rgba(170, 170, 170, 0.3)";
        }
      })
      .attr("stroke-width", (d: any) => {
        // Use weight-based thickness
        if (selected && (d.source.id === selected || d.target.id === selected)) {
          return getWeightStrokeWidth(d.weight);
        }
        // For non-selected edges, use thinner lines
        return Math.max(0.5, getWeightStrokeWidth(d.weight) * 0.3);
      })
      .attr("opacity", (d: any) => {
        if (selected) {
          return (d.source.id === selected || d.target.id === selected) ? 1 : 0.1;
        }
        // Show all edges when nothing is selected, but with lower opacity based on weight
        const absWeight = Math.abs(d.weight);
        return 0.3 + (absWeight * 0.4); // Opacity from 0.3 to 0.7 based on weight magnitude
      });

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

      // Highlight edges with weight-based visualization
      linkElems
        .attr("stroke", (d: any) => {
          if (d.source.id === selected || d.target.id === selected) {
            return getWeightColor(d.weight);
          }
          return "rgba(170, 170, 170, 0.3)";
        })
        .attr("stroke-width", (d: any) => {
          if (d.source.id === selected || d.target.id === selected) {
            return getWeightStrokeWidth(d.weight);
          }
          return Math.max(0.5, getWeightStrokeWidth(d.weight) * 0.3);
        })
        .attr("opacity", (d: any) => {
          if (d.source.id === selected || d.target.id === selected) {
            return 1;
          }
          return 0.1;
        });
    };

    updateHighlight();
  }, [modelWithWeights, selected]);

  return (
    <div style={{ display: "flex", width: "100%", height: "100%", overflow: "hidden" }}>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", padding: "20px" }}>
        <div style={{ marginBottom: "16px", display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <h2 style={{ margin: "0 0 8px 0", fontSize: "24px" }}>
              <TermDefinition term="neural network">Neural Network</TermDefinition> Visualization
            </h2>
            <p style={{ margin: 0, color: "#666", fontSize: "14px" }}>
              Click on <TermDefinition term="neuron">neurons</TermDefinition> to adjust <TermDefinition term="weight">weights</TermDefinition> and steer the <TermDefinition term="policy">policy</TermDefinition>
            </p>
          </div>
        <InfoPanel
          title="How to Use the Network Visualization & Weight Adjustment"
          content={
            <div>
              <p style={{ marginTop: 0 }}>
                <strong>Understanding the Visualization:</strong>
              </p>
              <ul style={{ paddingLeft: "20px", margin: "8px 0" }}>
                <li><strong>Circles</strong> represent <TermDefinition term="neuron">neurons</TermDefinition> (nodes) in the network</li>
                <li><strong>Lines</strong> represent <TermDefinition term="edge">edges</TermDefinition> (connections) with <TermDefinition term="weight">weights</TermDefinition></li>
                <li><strong>Blue edges</strong> = positive weights (strengthen connections)</li>
                <li><strong>Red edges</strong> = negative weights (weaken/invert connections)</li>
                <li><strong>Thicker edges</strong> = stronger <TermDefinition term="weight">weights</TermDefinition></li>
                <li><strong>Red nodes</strong> are selected neurons</li>
                <li><strong>Orange nodes</strong> are connected to the selected neuron</li>
              </ul>

              <p>
                <strong>Interactions:</strong>
              </p>
              <ul style={{ paddingLeft: "20px", margin: "8px 0" }}>
                <li><strong>Click a neuron</strong> to select it and open the weight adjustment panel</li>
                <li><strong>Adjust weights</strong> using sliders in the right panel to <TermDefinition term="steering">steer</TermDefinition> the <TermDefinition term="policy">policy</TermDefinition></li>
                <li><strong>Watch edges update</strong> in real-time as you change weights</li>
                <li><strong>Compare before/after</strong> values to see your changes</li>
                <li><strong>Hover over neurons</strong> to see their IDs</li>
                <li><strong>Scroll to zoom</strong> and <strong>drag to pan</strong> the visualization</li>
              </ul>

              <div style={{ marginTop: "16px", padding: "12px", background: "#e8f4f8", borderRadius: "4px" }}>
                <strong>💡 Steering the Policy:</strong>
                <p style={{ margin: "8px 0 0 0", fontSize: "13px" }}>
                  By adjusting <TermDefinition term="weight">weights</TermDefinition>, you can <TermDefinition term="steering">steer</TermDefinition> how the network behaves. 
                  This is called <TermDefinition term="perturbation">perturbation</TermDefinition> - making small changes to see how they affect the network's decisions. 
                  Positive weights strengthen connections, while negative weights weaken them.
                </p>
              </div>

              <div style={{ marginTop: "12px", padding: "12px", background: "#fff3cd", borderRadius: "4px" }}>
                <strong>🎯 Best Practices for Weight Adjustment:</strong>
                <ul style={{ margin: "8px 0 0 0", paddingLeft: "20px", fontSize: "13px" }}>
                  <li>Start with small <TermDefinition term="perturbation">perturbations</TermDefinition> to see gradual effects</li>
                  <li>Watch how edge thickness and color change as you adjust weights</li>
                  <li>Use the "Before/After" comparison to track your changes</li>
                  <li>Reset weights if you want to start over</li>
                  <li>Experiment with different neurons to understand their roles</li>
                </ul>
              </div>

              <div style={{ marginTop: "12px", padding: "12px", background: "#d4edda", borderRadius: "4px" }}>
                <strong>📚 Key Terms:</strong>
                <p style={{ margin: "8px 0 0 0", fontSize: "13px" }}>
                  Click on any underlined term (like <TermDefinition term="weight">weight</TermDefinition>, <TermDefinition term="steering">steering</TermDefinition>, or <TermDefinition term="policy">policy</TermDefinition>) 
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
          flex: 1,
          minHeight: "600px",
          overflow: "hidden"
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
              Adjust the <TermDefinition term="weight">weights</TermDefinition> in the panel on the right to see how changes affect the network. 
              <TermDefinition term="edge">Edges</TermDefinition> update in real-time: thicker and brighter lines indicate stronger connections.
            </p>
          </div>
        )}
      </div>

      {/* Weight Adjustment Panel */}
      <WeightAdjustmentPanel
        selectedNodeId={selected}
        weights={modelWithWeights?.weights || []}
        onWeightChange={handleWeightChange}
        originalWeights={originalWeights}
      />
    </div>
  );
}