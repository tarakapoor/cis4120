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
  const [showLabels, setShowLabels] = useState<boolean>(true);
  const [showNodeLabels, setShowNodeLabels] = useState<boolean>(false);
  const [nodeLabels, setNodeLabels] = useState<{[key: string]: string}>({});
  const [editingLabel, setEditingLabel] = useState<string | null>(null);
  
  // Initialize model with weights and store in state
  const [modelWithWeights, setModelWithWeights] = useState<ModelWithWeights | null>(null);
  const [originalWeights, setOriginalWeights] = useState<WeightData[]>([]);

  // Initialize weights when model changes
  useEffect(() => {
    if (model) {
      const initialized = initializeWeights(model);
      setModelWithWeights(initialized);
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

    const svgElement = ref.current as SVGSVGElement;
    if (!svgElement) return;
    
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
      .attr("stroke", "#4a90e2") // All edges blue
      .attr("stroke-width", 1)
      .attr("opacity", 0.15) // Very faded by default
      .style("cursor", "pointer")
      .on("click", (event, d) => {
        event.stopPropagation();
        setSelected((prev) => (prev === d.id ? null : d.id));
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
      .attr("opacity", 0.3) // Faded by default
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
      .on("click", (event, d) => {
        event.stopPropagation();
        setSelected((prev) => (prev === d.id ? null : d.id));
      });

    // --- HANDLE HIGHLIGHT LOGIC ---
    const updateHighlight = () => {
      if (!selected) {
        // Nothing selected - everything stays faded
        linkElems
          .attr("stroke", "#4a90e2") // All edges blue
          .attr("stroke-width", 1)
          .attr("opacity", 0.15);

        nodeElems
          .attr("fill", "#4a90e2")
          .attr("opacity", 0.3);
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
        .attr("opacity", (d: any) => (connectedNodeIds.has(d.id) ? 1 : 0.1));

      // Highlight edges - all blue, just varying opacity and thickness
      linkElems
        .attr("stroke", "#4a90e2") // All edges blue
        .attr("stroke-width", (d: any) => {
          if (d.source.id === selected || d.target.id === selected) {
            return getWeightStrokeWidth(d.weight);
          }
          return 1;
        })
        .attr("opacity", (d: any) => {
          if (d.source.id === selected || d.target.id === selected) {
            return 1;
          }
          return 0.05; // Very faded for non-connected edges
        });
    };

    updateHighlight();
    
    // Click background to deselect
    svg.on("click", () => setSelected(null));
    
    // --- DRAW LABELS (optional) ---
    if (showLabels) {
      // Layer labels
      const layerLabels = ['Input', ...Array(layers.length - 2).fill('Hidden'), 'Output'];
      
      g.selectAll("text.layer-label")
        .data(layers)
        .enter()
        .append("text")
        .attr("class", "layer-label")
        .attr("x", (d: any, i: number) => (i + 1) * layerSpacing)
        .attr("y", 30)
        .attr("text-anchor", "middle")
        .attr("font-family", "Inter, system-ui, -apple-system, sans-serif")
        .attr("font-size", "14px")
        .attr("font-weight", "600")
        .attr("fill", "#2c3e50")
        .text((d: any, i: number) => layerLabels[i]);
      
      // Feature count labels
      g.selectAll("text.feature-count")
        .data(layers)
        .enter()
        .append("text")
        .attr("class", "feature-count")
        .attr("x", (d: any, i: number) => (i + 1) * layerSpacing)
        .attr("y", 50)
        .attr("text-anchor", "middle")
        .attr("font-family", "Inter, system-ui, -apple-system, sans-serif")
        .attr("font-size", "12px")
        .attr("fill", "#7f8c8d")
        .text((d: any) => `${d.size} features`);
    }
    
    // --- DRAW NODE LABELS (optional) ---
    if (showNodeLabels) {
      // Filter to only input and output nodes
      const inputOutputNodes = nodes.filter((n: any) => 
        n.layer === 0 || n.layer === layers.length - 1
      );
      
      g.selectAll("text.node-label")
        .data(inputOutputNodes)
        .enter()
        .append("text")
        .attr("class", "node-label")
        .attr("x", (d: any) => {
          // Position labels to the left of input nodes, right of output nodes
          return d.layer === 0 ? d.x - 25 : d.x + 25;
        })
        .attr("y", (d: any) => d.y + 4)
        .attr("text-anchor", (d: any) => d.layer === 0 ? "end" : "start")
        .attr("font-family", "Inter, system-ui, -apple-system, sans-serif")
        .attr("font-size", "11px")
        .attr("fill", "#34495e")
        .style("cursor", "pointer")
        .text((d: any) => nodeLabels[d.id] || d.id)
        .on("click", (event, d) => {
          event.stopPropagation();
          setEditingLabel(d.id);
        });
    }
  }, [modelWithWeights, selected, showLabels, showNodeLabels, nodeLabels]);

  return (
    <div style={{ display: "flex", width: "100%", height: "100%", overflow: "hidden" }}>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", padding: "20px", overflow: "auto" }}>
        <div style={{ marginBottom: "16px", display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div style={{ flex: 1 }}>
            <h2 style={{ margin: "0 0 8px 0", fontSize: "24px", fontFamily: "Inter, system-ui, -apple-system, sans-serif", fontWeight: "600", color: "#2c3e50" }}>
              <TermDefinition term="neural network">Neural Network</TermDefinition> Visualization
            </h2>
            <p style={{ margin: 0, color: "#7f8c8d", fontSize: "14px", fontFamily: "Inter, system-ui, -apple-system, sans-serif" }}>
              Click on <TermDefinition term="neuron">neurons</TermDefinition> to adjust <TermDefinition term="weight">weights</TermDefinition> and steer the <TermDefinition term="policy">policy</TermDefinition>
            </p>
          </div>
          <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
            <button
              onClick={() => setShowLabels(!showLabels)}
              style={{
                padding: "8px 16px",
                background: showLabels ? "#4a90e2" : "white",
                color: showLabels ? "white" : "#4a90e2",
                border: "2px solid #4a90e2",
                borderRadius: "6px",
                cursor: "pointer",
                fontFamily: "Inter, system-ui, -apple-system, sans-serif",
                fontSize: "13px",
                fontWeight: "500",
                transition: "all 0.2s ease"
              }}
            >
              {showLabels ? "Hide" : "Show"} Layer Labels
            </button>
            <button
              onClick={() => setShowNodeLabels(!showNodeLabels)}
              style={{
                padding: "8px 16px",
                background: showNodeLabels ? "#8bc34a" : "white",
                color: showNodeLabels ? "white" : "#8bc34a",
                border: "2px solid #8bc34a",
                borderRadius: "6px",
                cursor: "pointer",
                fontFamily: "Inter, system-ui, -apple-system, sans-serif",
                fontSize: "13px",
                fontWeight: "500",
                transition: "all 0.2s ease"
              }}
            >
              {showNodeLabels ? "Hide" : "Show"} Node Labels
            </button>
        <InfoPanel
          title="How to Use the Network Visualization & Weight Adjustment"
          content={
            <div style={{ fontFamily: "Inter, system-ui, -apple-system, sans-serif" }}>
              <p style={{ marginTop: 0, lineHeight: "1.6" }}>
                <strong>Understanding the Visualization:</strong>
              </p>
              <ul style={{ paddingLeft: "20px", margin: "8px 0", lineHeight: "1.6" }}>
                <li><strong>Circles</strong> represent <TermDefinition term="neuron">neurons</TermDefinition> (nodes) in the network</li>
                <li><strong>Lines</strong> represent <TermDefinition term="edge">edges</TermDefinition> (connections) with <TermDefinition term="weight">weights</TermDefinition></li>
                <li><strong>Thicker edges</strong> indicate stronger <TermDefinition term="weight">weights</TermDefinition></li>
                <li><strong>Red nodes</strong> are selected neurons</li>
                <li><strong>Orange nodes</strong> are connected to the selected neuron</li>
              </ul>

              <p style={{ marginTop: "12px" }}>
                <strong>Interactions:</strong>
              </p>
              <ul style={{ paddingLeft: "20px", margin: "8px 0", lineHeight: "1.6" }}>
                <li><strong>Click a neuron</strong> to select it and open the weight adjustment panel</li>
                <li><strong>Click background</strong> to deselect and fade everything back</li>
                <li><strong>Click "Show Node Labels"</strong> to see input/output feature labels</li>
                <li><strong>Click on a node label</strong> to edit it (e.g., "Speed", "Torque")</li>
                <li><strong>Adjust weights</strong> using sliders in the right panel to <TermDefinition term="steering">steer</TermDefinition> the <TermDefinition term="policy">policy</TermDefinition></li>
                <li><strong>Watch edges update</strong> in real-time as you change weights</li>
                <li><strong>Compare before/after</strong> values to see your changes</li>
                <li><strong>Hover over neurons</strong> to see their IDs</li>
                <li><strong>Scroll to zoom</strong> and <strong>drag to pan</strong> the visualization</li>
              </ul>

              <div style={{ marginTop: "16px", padding: "16px", background: "#e3f2fd", borderRadius: "8px", borderLeft: "4px solid #2196f3" }}>
                <strong style={{ color: "#1976d2" }}>💡 Steering the Policy:</strong>
                <p style={{ margin: "8px 0 0 0", fontSize: "14px", lineHeight: "1.6" }}>
                  By adjusting <TermDefinition term="weight">weights</TermDefinition>, you can <TermDefinition term="steering">steer</TermDefinition> how the network behaves. 
                  This is called <TermDefinition term="perturbation">perturbation</TermDefinition> - making small changes to see how they affect the network's decisions. 
                  Positive weights strengthen connections, while negative weights weaken them.
                </p>
              </div>

              <div style={{ marginTop: "12px", padding: "16px", background: "#fff3cd", borderRadius: "8px", borderLeft: "4px solid #ffc107" }}>
                <strong style={{ color: "#f57c00" }}>🎯 Best Practices for Weight Adjustment:</strong>
                <ul style={{ margin: "8px 0 0 0", paddingLeft: "20px", fontSize: "14px", lineHeight: "1.6" }}>
                  <li>Start with small <TermDefinition term="perturbation">perturbations</TermDefinition> to see gradual effects</li>
                  <li>Watch how edge thickness and color change as you adjust weights</li>
                  <li>Use the "Before/After" comparison to track your changes</li>
                  <li>Reset weights if you want to start over</li>
                  <li>Experiment with different neurons to understand their roles</li>
                </ul>
              </div>

              <div style={{ marginTop: "12px", padding: "16px", background: "#f1f8e9", borderRadius: "8px", borderLeft: "4px solid #8bc34a" }}>
                <strong style={{ color: "#689f38" }}>📚 Key Terms:</strong>
                <p style={{ margin: "8px 0 0 0", fontSize: "14px", lineHeight: "1.6" }}>
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
        </div>

        <div style={{ 
          border: "1px solid #e0e0e0", 
          borderRadius: "12px", 
          background: "white",
          position: "relative",
          width: "100%",
          height: "600px",
          flexShrink: 0,
          overflow: "hidden",
          boxShadow: "0 2px 12px rgba(0,0,0,0.08)"
        }}>
          {tooltip && (
            <div
              style={{
                position: "fixed",
                top: tooltip.y + 10,
                left: tooltip.x + 10,
                background: "white",
                padding: "8px 12px",
                borderRadius: "6px",
                border: "2px solid #4a90e2",
                fontSize: "13px",
                fontFamily: "Inter, system-ui, -apple-system, sans-serif",
                fontWeight: "500",
                pointerEvents: "none",
                zIndex: 10,
                boxShadow: "0 4px 12px rgba(0,0,0,0.15)"
              }}
            >
              <strong style={{ color: "#2c3e50" }}>Neuron:</strong> <span style={{ color: "#4a90e2" }}>{tooltip.text}</span>
            </div>
          )}
          
          {editingLabel && (
            <div
              style={{
                position: "absolute",
                top: "50%",
                left: "50%",
                transform: "translate(-50%, -50%)",
                background: "white",
                padding: "24px",
                borderRadius: "12px",
                border: "2px solid #4a90e2",
                boxShadow: "0 8px 24px rgba(0,0,0,0.2)",
                zIndex: 100,
                minWidth: "300px",
                fontFamily: "Inter, system-ui, -apple-system, sans-serif"
              }}
            >
              <h4 style={{ margin: "0 0 16px 0", fontSize: "16px", fontWeight: "600", color: "#2c3e50" }}>
                Edit Node Label
              </h4>
              <p style={{ margin: "0 0 12px 0", fontSize: "13px", color: "#7f8c8d" }}>
                Node: <strong>{editingLabel}</strong>
              </p>
              <input
                type="text"
                placeholder="Enter label (e.g., 'Speed', 'X Position')"
                defaultValue={nodeLabels[editingLabel] || ""}
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    const value = (e.target as HTMLInputElement).value;
                    setNodeLabels(prev => ({...prev, [editingLabel]: value}));
                    setEditingLabel(null);
                  } else if (e.key === "Escape") {
                    setEditingLabel(null);
                  }
                }}
                style={{
                  width: "100%",
                  padding: "10px",
                  border: "2px solid #e0e0e0",
                  borderRadius: "6px",
                  fontSize: "14px",
                  fontFamily: "Inter, system-ui, -apple-system, sans-serif",
                  marginBottom: "12px",
                  outline: "none"
                }}
                onFocus={(e) => e.target.style.borderColor = "#4a90e2"}
                onBlur={(e) => e.target.style.borderColor = "#e0e0e0"}
              />
              <div style={{ display: "flex", gap: "8px", justifyContent: "flex-end" }}>
                <button
                  onClick={() => setEditingLabel(null)}
                  style={{
                    padding: "8px 16px",
                    background: "white",
                    color: "#7f8c8d",
                    border: "2px solid #e0e0e0",
                    borderRadius: "6px",
                    cursor: "pointer",
                    fontSize: "13px",
                    fontWeight: "500",
                    fontFamily: "Inter, system-ui, -apple-system, sans-serif"
                  }}
                >
                  Cancel
                </button>
                <button
                  onClick={(e) => {
                    const input = e.currentTarget.parentElement?.previousElementSibling as HTMLInputElement;
                    if (input) {
                      setNodeLabels(prev => ({...prev, [editingLabel]: input.value}));
                    }
                    setEditingLabel(null);
                  }}
                  style={{
                    padding: "8px 16px",
                    background: "#4a90e2",
                    color: "white",
                    border: "2px solid #4a90e2",
                    borderRadius: "6px",
                    cursor: "pointer",
                    fontSize: "13px",
                    fontWeight: "500",
                    fontFamily: "Inter, system-ui, -apple-system, sans-serif"
                  }}
                >
                  Save
                </button>
              </div>
              <p style={{ margin: "12px 0 0 0", fontSize: "12px", color: "#95a5a6", fontStyle: "italic" }}>
                Press Enter to save, Escape to cancel
              </p>
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
            marginBottom: "16px",
            padding: "16px",
            background: "linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%)",
            border: "2px solid #4a90e2",
            borderRadius: "8px",
            fontSize: "14px",
            fontFamily: "Inter, system-ui, -apple-system, sans-serif",
            flexShrink: 0
          }}>
            <strong style={{ color: "#2c3e50", fontSize: "15px" }}>Selected:</strong> <span style={{ color: "#4a90e2", fontWeight: "600" }}>{selected}</span>
            <p style={{ margin: "8px 0 0 0", fontSize: "13px", color: "#34495e", lineHeight: "1.5" }}>
              Adjust the <TermDefinition term="weight">weights</TermDefinition> in the panel on the right to see how changes affect the network. 
              <TermDefinition term="edge">Edges</TermDefinition> update in real-time: thicker and brighter lines indicate stronger connections.
            </p>
          </div>
        )}
        
        {showNodeLabels && modelWithWeights && (
          <div style={{
            marginTop: selected ? "0" : "16px",
            marginBottom: "16px",
            padding: "16px",
            background: "#f1f8e9",
            border: "2px solid #8bc34a",
            borderRadius: "8px",
            fontSize: "13px",
            fontFamily: "Inter, system-ui, -apple-system, sans-serif",
            flexShrink: 0
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
              <strong style={{ color: "#2c3e50", fontSize: "14px" }}>💡 Node Labeling Tips:</strong>
            </div>
            <p style={{ margin: "0 0 8px 0", color: "#34495e", lineHeight: "1.5" }}>
              Click on any input or output node label to customize it. Give meaningful names like "Velocity", "Position", "Action 1", etc.
            </p>
            <details style={{ marginTop: "8px" }}>
              <summary style={{ cursor: "pointer", color: "#689f38", fontWeight: "500", userSelect: "none" }}>
                Quick Label All Nodes
              </summary>
              <div style={{ marginTop: "12px", padding: "12px", background: "white", borderRadius: "6px" }}>
                <p style={{ margin: "0 0 8px 0", fontSize: "12px", color: "#7f8c8d" }}>
                  Enter comma-separated labels for all input nodes, then output nodes:
                </p>
                <input
                  type="text"
                  placeholder="e.g., pos_x, pos_y, vel_x, vel_y, angle"
                  style={{
                    width: "100%",
                    padding: "8px",
                    border: "1px solid #e0e0e0",
                    borderRadius: "4px",
                    fontSize: "12px",
                    fontFamily: "Inter, system-ui, -apple-system, sans-serif",
                    marginBottom: "8px"
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      const value = (e.target as HTMLInputElement).value;
                      const labels = value.split(",").map(s => s.trim()).filter(s => s);
                      const inputLayer = modelWithWeights.layers[0];
                      const outputLayer = modelWithWeights.layers[modelWithWeights.layers.length - 1];
                      const newLabels: {[key: string]: string} = {};
                      
                      // Label input nodes
                      for (let i = 0; i < Math.min(labels.length, inputLayer.size); i++) {
                        newLabels[`L0-N${i}`] = labels[i];
                      }
                      
                      // Label output nodes if there are more labels
                      const outputStart = inputLayer.size;
                      for (let i = 0; i < Math.min(labels.length - outputStart, outputLayer.size); i++) {
                        newLabels[`L${modelWithWeights.layers.length - 1}-N${i}`] = labels[outputStart + i];
                      }
                      
                      setNodeLabels(prev => ({...prev, ...newLabels}));
                      (e.target as HTMLInputElement).value = "";
                    }
                  }}
                />
                <p style={{ margin: "0", fontSize: "11px", color: "#95a5a6", fontStyle: "italic" }}>
                  Press Enter to apply. First {modelWithWeights.layers[0].size} labels → inputs, 
                  next {modelWithWeights.layers[modelWithWeights.layers.length - 1].size} → outputs
                </p>
              </div>
            </details>
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