import React, { useState, useEffect } from "react";
import TermDefinition from "../UI/TermDefinition";
import InfoPanel from "../UI/InfoPanel";
import { WeightData, getOutgoingWeights, updateWeight } from "../../utils/modelUtils";

interface WeightAdjustmentPanelProps {
  selectedNodeId: string | null;
  weights: WeightData[];
  onWeightChange: (weights: WeightData[]) => void;
  originalWeights: WeightData[]; // Store original weights for before/after comparison
}

export default function WeightAdjustmentPanel({
  selectedNodeId,
  weights,
  onWeightChange,
  originalWeights
}: WeightAdjustmentPanelProps) {
  const [localWeights, setLocalWeights] = useState<Record<string, number>>({});

  // Get outgoing weights for the selected node
  const outgoingWeights = selectedNodeId 
    ? getOutgoingWeights(weights, selectedNodeId)
    : [];

  // Initialize local weights when selection changes
  useEffect(() => {
    if (selectedNodeId) {
      const outgoing = getOutgoingWeights(weights, selectedNodeId);
      const weightMap: Record<string, number> = {};
      outgoing.forEach(w => {
        weightMap[w.targetId] = w.weight;
      });
      setLocalWeights(weightMap);
    } else {
      setLocalWeights({});
    }
  }, [selectedNodeId, weights]);

  const handleWeightChange = (targetId: string, newWeight: number) => {
    if (!selectedNodeId) return;

    // Update local state
    setLocalWeights(prev => ({
      ...prev,
      [targetId]: newWeight
    }));

    // Update global weights
    const updatedWeights = updateWeight(weights, selectedNodeId, targetId, newWeight);
    onWeightChange(updatedWeights);
  };

  const getOriginalWeight = (targetId: string): number => {
    if (!selectedNodeId) return 0;
    const original = originalWeights.find(
      w => w.sourceId === selectedNodeId && w.targetId === targetId
    );
    return original ? original.weight : 0;
  };

  const resetWeight = (targetId: string) => {
    const original = getOriginalWeight(targetId);
    handleWeightChange(targetId, original);
  };

  const resetAllWeights = () => {
    if (!selectedNodeId) return;
    outgoingWeights.forEach(w => {
      const original = getOriginalWeight(w.targetId);
      handleWeightChange(w.targetId, original);
    });
  };

  if (!selectedNodeId || outgoingWeights.length === 0) {
    return (
      <div style={{
        width: "384px",
        padding: "20px",
        paddingBottom: "40px",
        borderLeft: "1px solid #e2e8f0",
        background: "#ffffff",
        height: "100%",
        overflowY: "auto",
        boxSizing: "border-box"
      }}>
        <h3 style={{ margin: "0 0 16px 0", fontSize: "18px", fontWeight: "600", color: "#1e293b" }}>
          Weight Adjustment
        </h3>
        <p style={{ color: "#64748b", fontSize: "14px", lineHeight: "1.6" }}>
          Select a <TermDefinition term="neuron">neuron</TermDefinition> in the network to adjust its outgoing <TermDefinition term="weight">weights</TermDefinition>.
        </p>
        <div style={{
          marginTop: "20px",
          padding: "12px",
          background: "#eff6ff",
          borderRadius: "4px",
          fontSize: "13px",
          flexShrink: 0,
          border: "1px solid #bfdbfe"
        }}>
          <strong style={{ color: "#1e40af" }}>💡 Tip:</strong>
          <p style={{ margin: "8px 0 0 0", color: "#475569" }}>
            Click on any <TermDefinition term="neuron">neuron</TermDefinition> (circle) in the visualization to see and adjust its <TermDefinition term="connection">connections</TermDefinition> to other neurons.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div style={{
      width: "384px",
      padding: "20px",
      paddingBottom: "40px",
      borderLeft: "1px solid #e2e8f0",
      background: "#ffffff",
      height: "100%",
      overflowY: "auto",
      boxSizing: "border-box",
      // flexShrink: 0
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
        <h3 style={{ margin: 0, fontSize: "18px", fontWeight: "600", color: "#1e293b" }}>
          Adjust <TermDefinition term="weight">Weights</TermDefinition>
        </h3>
        <InfoPanel
          title="How to Adjust Weights"
          content={
            <div>
              <p style={{ marginTop: 0 }}>
                <strong>What are Weights?</strong>
              </p>
              <p>
                <TermDefinition term="weight">Weights</TermDefinition> control how strongly one <TermDefinition term="neuron">neuron</TermDefinition> influences another. 
                By adjusting these values, you can "steer" the <TermDefinition term="policy">policy</TermDefinition> and change how the network behaves.
              </p>

              <p>
                <strong>How to Use:</strong>
              </p>
              <ul style={{ paddingLeft: "20px", margin: "8px 0" }}>
                <li>Use the sliders to adjust <TermDefinition term="weight">weight</TermDefinition> values</li>
                <li>Positive weights (blue) strengthen the connection</li>
                <li>Negative weights (red) weaken or invert the connection</li>
                <li>Watch the visualization update in real-time</li>
                <li>Compare "Before" and "After" values to see changes</li>
              </ul>

              <div style={{ marginTop: "16px", padding: "12px", background: "#f8fafc", borderRadius: "4px", border: "1px solid #e2e8f0" }}>
                <strong style={{ color: "#475569" }}>🎯 Best Practices:</strong>
                <ul style={{ margin: "8px 0 0 0", paddingLeft: "20px", fontSize: "13px" }}>
                  <li>Make small adjustments first to see the effects</li>
                  <li>Positive weights (0 to 1) enhance signals</li>
                  <li>Negative weights (-1 to 0) suppress or invert signals</li>
                  <li>Use "Reset" to restore original values</li>
                  <li>Thicker, brighter edges indicate stronger connections</li>
                </ul>
              </div>

              <div style={{ marginTop: "12px", padding: "12px", background: "#eff6ff", borderRadius: "4px", border: "1px solid #bfdbfe" }}>
                <strong style={{ color: "#1e40af" }}>💡 Understanding the Visualization:</strong>
                <p style={{ margin: "8px 0 0 0", fontSize: "13px" }}>
                  As you adjust <TermDefinition term="weight">weights</TermDefinition>, the <TermDefinition term="edge">edges</TermDefinition> (lines) in the network change:
                  <ul style={{ margin: "8px 0 0 0", paddingLeft: "20px", fontSize: "13px" }}>
                    <li><strong>Color:</strong> 
                      <ul style={{ marginTop: "4px", paddingLeft: "16px" }}>
                        <li><strong>Blue edges</strong> = positive weights (strengthen connections)</li>
                        <li><strong>Grey edges</strong> = negative weights (weaken/invert connections)</li>
                      </ul>
                    </li>
                    <li><strong>Thickness:</strong> Thicker edges = larger weight magnitude (stronger absolute values)</li>
                    <li><strong>Shade:</strong> Darker shades (for both blue and grey) = larger absolute weight values</li>
                  </ul>
                </p>
              </div>
            </div>
          }
          position="bottom-left"
          size="large"
        />
      </div>

      <div style={{
        marginBottom: "16px",
        padding: "12px",
        background: "#f8fafc",
        border: "1px solid #e2e8f0",
        borderRadius: "4px"
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
          <strong style={{ fontSize: "14px", color: "#1e293b" }}>Selected Node:</strong>
          <span style={{ fontFamily: "monospace", fontSize: "13px", color: "#475569" }}>{selectedNodeId}</span>
        </div>
        <div style={{ fontSize: "12px", color: "#64748b" }}>
          Adjusting {outgoingWeights.length} outgoing <TermDefinition term="connection">{outgoingWeights.length !== 1 ? "connections" : "connection"}</TermDefinition>
        </div>
      </div>

      <div style={{ marginBottom: "16px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <button
          onClick={resetAllWeights}
          style={{
            padding: "8px 16px",
            background: "#64748b",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
            fontSize: "13px",
            fontWeight: "500",
            transition: "all 0.2s"
          }}
        >
          Reset All to Original
        </button>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
        {outgoingWeights.map((weightData) => {
          const currentWeight = localWeights[weightData.targetId] ?? weightData.weight;
          const originalWeight = getOriginalWeight(weightData.targetId);
          const hasChanged = Math.abs(currentWeight - originalWeight) > 0.001;

          return (
            <div
              key={weightData.targetId}
              style={{
                padding: "16px",
                background: hasChanged ? "#eff6ff" : "#ffffff",
                border: hasChanged ? "2px solid #2563eb" : "1px solid #e2e8f0",
                borderRadius: "6px"
              }}
            >
              <div style={{ marginBottom: "12px" }}>
                <div style={{ fontSize: "13px", fontWeight: "500", marginBottom: "4px", color: "#1e293b" }}>
                  To: <span style={{ fontFamily: "monospace", color: "#2563eb" }}>{weightData.targetId}</span>
                </div>
              </div>

              <div style={{ marginBottom: "12px" }}>
                <label style={{ display: "block", marginBottom: "8px", fontSize: "13px", fontWeight: "500" }}>
                  <TermDefinition term="weight">Weight</TermDefinition> Value:
                </label>
                <input
                  type="range"
                  min="-1"
                  max="1"
                  step="0.01"
                  value={currentWeight}
                  onChange={(e) => handleWeightChange(weightData.targetId, parseFloat(e.target.value))}
                  style={{
                    width: "100%",
                    height: "6px",
                    borderRadius: "3px",
                    outline: "none",
                    cursor: "pointer"
                  }}
                />
                <div style={{ display: "flex", justifyContent: "space-between", marginTop: "4px", fontSize: "11px", color: "#64748b" }}>
                  <span>-1.0</span>
                  <span>0.0</span>
                  <span>1.0</span>
                </div>
              </div>

              <div style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "12px",
                marginBottom: "12px",
                fontSize: "12px"
              }}>
                <div style={{
                  padding: "8px",
                  background: "#f8fafc",
                  borderRadius: "4px",
                  border: "1px solid #e2e8f0"
                }}>
                  <div style={{ color: "#64748b", marginBottom: "4px" }}>Before:</div>
                  <div style={{
                    fontFamily: "monospace",
                    fontSize: "14px",
                    fontWeight: "600",
                    color: originalWeight >= 0 ? "#2563eb" : "#475569"
                  }}>
                    {originalWeight.toFixed(3)}
                  </div>
                </div>
                <div style={{
                  padding: "8px",
                  background: hasChanged ? "#eff6ff" : "#f8fafc",
                  borderRadius: "4px",
                  border: hasChanged ? "2px solid #2563eb" : "1px solid #e2e8f0"
                }}>
                  <div style={{ color: "#64748b", marginBottom: "4px" }}>After:</div>
                  <div style={{
                    fontFamily: "monospace",
                    fontSize: "14px",
                    fontWeight: "600",
                    color: currentWeight >= 0 ? "#2563eb" : "#475569"
                  }}>
                    {currentWeight.toFixed(3)}
                  </div>
                </div>
              </div>

              <div style={{
                fontSize: "11px",
                color: "#64748b",
                marginBottom: "12px",
                padding: "6px",
                background: "#f8fafc",
                borderRadius: "4px"
              }}>
                <strong>Change:</strong>{" "}
                <span style={{
                  color: hasChanged ? (currentWeight > originalWeight ? "#2563eb" : "#475569") : "#64748b",
                  fontFamily: "monospace"
                }}>
                  {currentWeight > originalWeight ? "+" : ""}
                  {(currentWeight - originalWeight).toFixed(3)}
                </span>
              </div>

              <button
                onClick={() => resetWeight(weightData.targetId)}
                disabled={!hasChanged}
                style={{
                  width: "100%",
                  padding: "6px 12px",
                  background: hasChanged ? "#64748b" : "#e2e8f0",
                  color: hasChanged ? "white" : "#94a3b8",
                  border: "none",
                  borderRadius: "4px",
                  cursor: hasChanged ? "pointer" : "not-allowed",
                  fontSize: "12px",
                  opacity: hasChanged ? 1 : 0.6,
                  transition: "all 0.2s"
                }}
              >
                Reset This Weight
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}

