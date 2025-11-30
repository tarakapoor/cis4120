// ModelUpload.tsx
import React, { useState } from "react";
import InfoPanel from "../UI/InfoPanel";
import TermDefinition from "../UI/TermDefinition";

type Summary = { layers: { size: number }[] };
type Props = {
  onLoadModel: (summary: Summary) => void;
  onRunComplete?: (results: any) => void;
  isMinimized?: boolean;
  onToggleMinimize?: () => void;
};

export default function ModelUpload({ onLoadModel, onRunComplete, isMinimized = false, onToggleMinimize }: Props) {
  const [modelId, setModelId] = useState<string | null>(null);
  const [weightKeys, setWeightKeys] = useState<string[]>([]);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [runResults, setRunResults] = useState<any>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [environment, setEnvironment] = useState("Walker2d-v4");
  const [captureActivations, setCaptureActivations] = useState(true);

  // SAE state
  const [saeLoaded, setSaeLoaded] = useState(false);
  const [saeInfo, setSaeInfo] = useState<any>(null);
  const [topFeatures, setTopFeatures] = useState<any[]>([]);
  const [selectedFeature, setSelectedFeature] = useState<number | null>(null);
  const [featureAlpha, setFeatureAlpha] = useState(2.0);

  async function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    const lower = file.name.toLowerCase();
    if (lower.endsWith(".json")) {
      // Old behavior: parse JSON client-side
      const text = await file.text();
      try {
        const json = JSON.parse(text);
        onLoadModel(json);
        setModelId(null);
        setWeightKeys([]);
      } catch (err) {
        console.error("Invalid JSON", err);
        alert("Invalid JSON file.");
      }
      return;
    }

    if (lower.endsWith(".pt")) {
      // New behavior: send to backend
      try {
        setBusy(true);
        const form = new FormData();
        form.append("file", file);
        const res = await fetch("http://localhost:8000/upload", {
          method: "POST",
          body: form,
        });
        if (!res.ok) {
          const msg = await res.text();
          throw new Error(msg);
        }
        const data = await res.json();
        setModelId(data.model_id);
        onLoadModel(data.summary); // same shape your UI already expects
        // grab keys list for perturbations
        const sumRes = await fetch(`http://localhost:8000/model/${data.model_id}/summary`);
        const sumData = await sumRes.json();
        setWeightKeys(sumData.keys || []);
        setSelectedKey(sumData.keys?.find((k: string) => k.endsWith(".weight")) || null);
      } catch (e: any) {
        console.error(e);
        alert(`Upload failed: ${e?.message || e}`);
      } finally {
        setBusy(false);
      }
      return;
    }

    alert("Please upload a .json or .pt file");
  }

  async function perturb(op: "scale" | "add_noise" | "set") {
    if (!modelId) return alert("No .pt model loaded");
    const key = selectedKey;
    const body: any = { op, key };
    if (op === "scale") {
      const s = prompt("Scale factor (e.g., 0.9):", "0.9");
      if (!s) return;
      body.scale = parseFloat(s);
    } else if (op === "add_noise") {
      const std = prompt("Noise std (e.g., 0.01):", "0.01");
      if (!std) return;
      body.std = parseFloat(std);
    } else if (op === "set") {
      const v = prompt("Set all weights to value:", "0.0");
      if (!v) return;
      body.value = parseFloat(v);
    }
    try {
      setBusy(true);
      const res = await fetch(`http://localhost:8000/model/${modelId}/perturb`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(await res.text());
      await res.json();
      alert("Weights perturbed successfully! You can now run the model to see the effect.");
      // Optional: re-pull summary if you want to reflect width changes (usually unchanged)
    } catch (e: any) {
      alert(`Perturbation failed: ${e?.message || e}`);
    } finally {
      setBusy(false);
    }
  }

  async function runModel() {
    if (!modelId) return alert("No .pt model loaded");

    try {
      setIsRunning(true);
      setRunResults(null);

      const res = await fetch(`http://localhost:8000/model/${modelId}/save_and_run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          script: "rollout",
          env: environment,
          num_traj: 5,
          max_steps: 300,
          capture_activations: captureActivations
        }),
      });

      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();

      if (data.returncode !== 0) {
        alert(`Model run failed with return code ${data.returncode}. Check console for details.`);
        console.error("STDERR:", data.stderr);
        console.log("STDOUT:", data.stdout);
      } else {
        setRunResults(data.results);
        if (onRunComplete && data.results) {
          onRunComplete(data.results);
        }
      }
    } catch (e: any) {
      alert(`Failed to run model: ${e?.message || e}`);
      console.error(e);
    } finally {
      setIsRunning(false);
    }
  }

  // SAE Functions
  async function loadSAE() {
    if (!modelId) return alert("No model loaded");
    const artifactsDir = prompt("Enter artifacts directory path (or '.' for python/ directory):", ".");
    if (artifactsDir === null) return;

    try {
      setBusy(true);
      const res = await fetch(`http://localhost:8000/model/${modelId}/load_sae`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ artifacts_dir: artifactsDir, tap_index: 4 })
      });

      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();

      setSaeLoaded(true);
      setSaeInfo(data);
      alert(`SAE loaded! d_latent=${data.d_latent}, k=${data.k}`);

      // Auto-interpret features
      if (data.has_cached_data) {
        await interpretFeatures();
      }
    } catch (e: any) {
      alert(`Failed to load SAE: ${e?.message || e}`);
      console.error(e);
    } finally {
      setBusy(false);
    }
  }

  async function interpretFeatures(targetDim: number = 0) {
    if (!modelId || !saeLoaded) return;

    try {
      setBusy(true);
      const res = await fetch(`http://localhost:8000/model/${modelId}/interpret_features`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_dim: targetDim, top_k: 10 })
      });

      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();

      setTopFeatures(data.features);
      if (data.features.length > 0) {
        setSelectedFeature(data.features[0].feature_idx);
      }
    } catch (e: any) {
      console.error("Failed to interpret features:", e);
    } finally {
      setBusy(false);
    }
  }

  async function applyFeaturePerturbation() {
    if (!modelId || !saeLoaded || selectedFeature === null) return;

    try {
      setBusy(true);
      const res = await fetch(`http://localhost:8000/model/${modelId}/sae_perturb`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ feature_idx: selectedFeature, alpha: featureAlpha })
      });

      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();

      alert(`Feature ${selectedFeature} perturbed with α=${featureAlpha.toFixed(2)}\nDecoder norm: ${data.decoder_norm.toFixed(3)}`);
    } catch (e: any) {
      alert(`Failed to perturb feature: ${e?.message || e}`);
      console.error(e);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ 
      width: isMinimized ? "40px" : "280px", 
      padding: isMinimized ? "24px 8px" : "24px 20px", 
      borderRight: "1px solid rgba(71, 85, 105, 0.3)", 
      overflowY: "auto", 
      height: "100vh", 
      background: "#1a2332", 
      display: "flex", 
      flexDirection: "column",
      transition: "width 0.3s ease, padding 0.3s ease",
      position: "relative"
    }}>
      {/* Collapse button on the edge */}
      <button
        onClick={onToggleMinimize}
        style={{
          position: "absolute",
          top: "20px",
          right: "-20px",
          padding: "12px 6px",
          background: "#2d3748",
          border: "1px solid rgba(71, 85, 105, 0.5)",
          borderRadius: isMinimized ? "0 8px 8px 0" : "8px 0 0 8px",
          cursor: "pointer",
          fontSize: "18px",
          color: "#94a3b8",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          width: "32px",
          height: "60px",
          transition: "all 0.2s",
          boxShadow: "0 2px 8px rgba(0,0,0,0.3)",
          zIndex: 100,
          borderRight: isMinimized ? "1px solid rgba(71, 85, 105, 0.5)" : "none",
          margin: 0,
          lineHeight: 1
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = "#475569";
          e.currentTarget.style.color = "#e2e8f0";
          e.currentTarget.style.boxShadow = "0 4px 12px rgba(0,0,0,0.4)";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = "#2d3748";
          e.currentTarget.style.color = "#94a3b8";
          e.currentTarget.style.boxShadow = "0 2px 8px rgba(0,0,0,0.3)";
        }}
        title={isMinimized ? "Expand panel" : "Minimize panel"}
      >
        {isMinimized ? "▶" : "◀"}
      </button>

      <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "20px", justifyContent: isMinimized ? "center" : "flex-start" }}>
        {!isMinimized && (
          <>
            <h3 style={{ margin: 0, fontSize: "18px", fontWeight: "600", color: "#e2e8f0" }}>Upload <TermDefinition term="neural network">Model</TermDefinition></h3>
            <InfoPanel
              title="How to Upload a Model"
              content={
                <div>
                  <p><strong>New:</strong> You can now upload <code>.pt</code> PyTorch checkpoints. We'll infer layer sizes and convert to the JSON structure your graph needs.</p>
                  {/* keep the rest of your existing help content if useful */}
                </div>
              }
              position="bottom-right"
              size="large"
            />
          </>
        )}
      </div>

      {!isMinimized && (
        <>
          <div style={{ marginBottom: "20px", width: "100%" }}>
            <label style={{ display: "block", marginBottom: "10px", fontSize: "14px", fontWeight: 500, color: "#e2e8f0", textAlign: "left" }}>
              Select Model File (.json or .pt):
            </label>
            <input
              type="file"
              accept=".json,.pt"
              onChange={handleFile}
              disabled={busy}
              style={{ width: "100%", padding: "10px", border: "1px solid rgba(71, 85, 105, 0.5)", borderRadius: 6, fontSize: 14, background: "#2d3748", color: "#e2e8f0", boxSizing: "border-box" }}
            />
          </div>

          {modelId && (
            <>
              <div style={{ padding: "16px", background: "#0f172a", borderRadius: 6, marginBottom: "16px", border: "1px solid rgba(71, 85, 105, 0.3)", width: "100%", boxSizing: "border-box" }}>
            <div style={{ fontWeight: 600, marginBottom: "12px", color: "#e2e8f0", fontSize: "15px" }}>Weight editing</div>
            <label style={{ display: "block", marginBottom: "8px", fontSize: 13, color: "#94a3b8", textAlign: "left" }}>Choose tensor key:</label>
            <select
              value={selectedKey ?? ""}
              onChange={(e) => setSelectedKey(e.target.value)}
              style={{ width: "100%", marginBottom: "12px", padding: "8px", border: "1px solid rgba(71, 85, 105, 0.5)", borderRadius: 6, background: "#2d3748", color: "#e2e8f0", fontSize: 13, boxSizing: "border-box" }}
            >
              {weightKeys.map(k => <option key={k} value={k}>{k}</option>)}
            </select>

            <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", justifyContent: "flex-start" }}>
              <button 
                onClick={() => perturb("scale")} 
                disabled={busy}
                style={{
                  padding: "8px 14px",
                  background: busy ? "#475569" : "#475569",
                  color: "white",
                  border: "none",
                  borderRadius: 6,
                  cursor: busy ? "not-allowed" : "pointer",
                  fontSize: 13,
                  fontWeight: 500,
                  transition: "all 0.2s",
                  boxShadow: busy ? "none" : "0 2px 4px rgba(0,0,0,0.3)"
                }}
                onMouseEnter={(e) => {
                  if (!busy) {
                    e.currentTarget.style.background = "#64748b";
                    e.currentTarget.style.transform = "translateY(-1px)";
                  }
                }}
                onMouseLeave={(e) => {
                  if (!busy) {
                    e.currentTarget.style.background = "#475569";
                    e.currentTarget.style.transform = "translateY(0)";
                  }
                }}
              >
                Scale
              </button>
              <button 
                onClick={() => perturb("add_noise")} 
                disabled={busy}
                style={{
                  padding: "8px 14px",
                  background: busy ? "#475569" : "#475569",
                  color: "white",
                  border: "none",
                  borderRadius: 6,
                  cursor: busy ? "not-allowed" : "pointer",
                  fontSize: 13,
                  fontWeight: 500,
                  transition: "all 0.2s",
                  boxShadow: busy ? "none" : "0 2px 4px rgba(0,0,0,0.3)"
                }}
                onMouseEnter={(e) => {
                  if (!busy) {
                    e.currentTarget.style.background = "#64748b";
                    e.currentTarget.style.transform = "translateY(-1px)";
                  }
                }}
                onMouseLeave={(e) => {
                  if (!busy) {
                    e.currentTarget.style.background = "#475569";
                    e.currentTarget.style.transform = "translateY(0)";
                  }
                }}
              >
                Add noise
              </button>
              <button 
                onClick={() => perturb("set")} 
                disabled={busy}
                style={{
                  padding: "8px 14px",
                  background: busy ? "#475569" : "#475569",
                  color: "white",
                  border: "none",
                  borderRadius: 6,
                  cursor: busy ? "not-allowed" : "pointer",
                  fontSize: 13,
                  fontWeight: 500,
                  transition: "all 0.2s",
                  boxShadow: busy ? "none" : "0 2px 4px rgba(0,0,0,0.3)"
                }}
                onMouseEnter={(e) => {
                  if (!busy) {
                    e.currentTarget.style.background = "#64748b";
                    e.currentTarget.style.transform = "translateY(-1px)";
                  }
                }}
                onMouseLeave={(e) => {
                  if (!busy) {
                    e.currentTarget.style.background = "#475569";
                    e.currentTarget.style.transform = "translateY(0)";
                  }
                }}
              >
                Set value
              </button>
            </div>

            <p style={{ marginTop: 10, fontSize: 12, color: "#94a3b8" }}>
              Changes are in-memory. Perturbations will affect the model run below.
            </p>
          </div>

          <div style={{ padding: "16px", background: "#0f172a", borderRadius: 6, border: "1px solid rgba(71, 85, 105, 0.3)", width: "100%", boxSizing: "border-box" }}>
            <div style={{ fontWeight: 600, marginBottom: "12px", color: "#e2e8f0", fontSize: "15px" }}>Run Model</div>

            <label style={{ display: "block", marginBottom: "8px", fontSize: 13, color: "#94a3b8", textAlign: "left" }}>Environment:</label>
            <select
              value={environment}
              onChange={(e) => setEnvironment(e.target.value)}
              style={{ width: "100%", marginBottom: "12px", padding: "8px", border: "1px solid rgba(71, 85, 105, 0.5)", borderRadius: 6, background: "#2d3748", color: "#e2e8f0", fontSize: 13, boxSizing: "border-box" }}
            >
              <option value="Walker2d-v4">Walker2d-v4</option>
              <option value="HalfCheetah-v4">HalfCheetah-v4</option>
              <option value="Hopper-v4">Hopper-v4</option>
              <option value="Ant-v4">Ant-v4</option>
              <option value="hard_stable">hard_stable (6-dim obs)</option>
            </select>

            <div style={{ marginBottom: "12px" }}>
              <label style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: 13, cursor: "pointer", textAlign: "left", color: "#94a3b8" }}>
                <input
                  type="checkbox"
                  checked={captureActivations}
                  onChange={(e) => setCaptureActivations(e.target.checked)}
                  style={{ cursor: "pointer" }}
                />
                Capture activations (enables visualization)
              </label>
            </div>

            <p style={{ fontSize: 13, marginBottom: "12px", color: "#94a3b8", textAlign: "left" }}>
              Test the current model (with any perturbations) in the selected environment.
            </p>
            <button
              onClick={runModel}
              disabled={isRunning}
              style={{
                width: "100%",
                padding: "10px 16px",
                background: isRunning ? "#475569" : "#475569",
                color: "white",
                border: "none",
                borderRadius: 6,
                cursor: isRunning ? "not-allowed" : "pointer",
                fontSize: 14,
                fontWeight: 600,
                transition: "all 0.2s",
                boxShadow: isRunning ? "none" : "0 2px 6px rgba(0,0,0,0.3)"
              }}
              onMouseEnter={(e) => {
                if (!isRunning) {
                  e.currentTarget.style.background = "#64748b";
                  e.currentTarget.style.transform = "translateY(-1px)";
                }
              }}
              onMouseLeave={(e) => {
                if (!isRunning) {
                  e.currentTarget.style.background = "#475569";
                  e.currentTarget.style.transform = "translateY(0)";
                }
              }}
            >
              {isRunning ? "Running..." : "Run Model"}
              </button>
            </div>

            {runResults && (
              <div style={{ marginTop: "16px", padding: "16px", background: "#0f172a", borderRadius: 6, border: "1px solid rgba(71, 85, 105, 0.3)", width: "100%", boxSizing: "border-box" }}>
                <div style={{ fontWeight: 600, marginBottom: "12px", color: "#e2e8f0", fontSize: "15px" }}>Results</div>
                <div style={{ fontSize: 13, lineHeight: 1.6, textAlign: "left", color: "#94a3b8" }}>
                  <div><strong style={{ color: "#e2e8f0" }}>Trajectories:</strong> {runResults.num_trajectories}</div>
                  <div><strong style={{ color: "#e2e8f0" }}>Avg Reward:</strong> {runResults.avg_reward?.toFixed(2)}</div>
                  <div><strong style={{ color: "#e2e8f0" }}>Max Reward:</strong> {runResults.max_reward?.toFixed(2)}</div>
                  <div><strong style={{ color: "#e2e8f0" }}>Min Reward:</strong> {runResults.min_reward?.toFixed(2)}</div>
                  <div><strong style={{ color: "#e2e8f0" }}>Avg Length:</strong> {runResults.avg_length?.toFixed(1)} steps</div>
                </div>
              </div>
            )}

            {/* SAE Section */}
            <div style={{ marginTop: "16px", padding: "16px", background: "#0f172a", borderRadius: 6, border: "1px solid rgba(71, 85, 105, 0.3)", width: "100%", boxSizing: "border-box" }}>
            <div style={{ fontWeight: 600, marginBottom: "12px", color: "#e2e8f0", fontSize: "15px" }}>SAE Feature Analysis</div>

            {!saeLoaded ? (
              <>
                <p style={{ fontSize: 13, marginBottom: "12px", color: "#94a3b8", textAlign: "left" }}>
                  Load a Sparse Autoencoder to interpret and perturb learned features.
                </p>
                <button
                  onClick={loadSAE}
                  disabled={busy}
                  style={{
                    width: "100%",
                    padding: "10px 16px",
                    background: busy ? "#475569" : "#475569",
                    color: "white",
                    border: "none",
                    borderRadius: 6,
                    cursor: busy ? "not-allowed" : "pointer",
                    fontSize: 14,
                    fontWeight: 600,
                    transition: "all 0.2s",
                    boxShadow: busy ? "none" : "0 2px 6px rgba(0,0,0,0.3)"
                  }}
                  onMouseEnter={(e) => {
                    if (!busy) {
                      e.currentTarget.style.background = "#64748b";
                      e.currentTarget.style.transform = "translateY(-1px)";
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!busy) {
                      e.currentTarget.style.background = "#475569";
                      e.currentTarget.style.transform = "translateY(0)";
                    }
                  }}
                >
                  {busy ? "Loading..." : "Load SAE"}
                </button>
              </>
            ) : (
              <>
                <div style={{ fontSize: 13, marginBottom: "12px", color: "#94a3b8", textAlign: "left" }}>
                  <div>Latent dim: {saeInfo?.d_latent}, Top-K: {saeInfo?.k}</div>
                </div>

                {topFeatures.length > 0 && (
                  <div style={{ marginBottom: "12px" }}>
                    <label style={{ display: "block", marginBottom: "8px", fontSize: 13, textAlign: "left", color: "#94a3b8" }}>
                      Top Interpretable Features:
                    </label>
                    <select
                      value={selectedFeature ?? ""}
                      onChange={(e) => setSelectedFeature(parseInt(e.target.value))}
                      style={{ width: "100%", marginBottom: "12px", fontSize: 13, padding: "8px", border: "1px solid rgba(71, 85, 105, 0.5)", borderRadius: 6, background: "#2d3748", color: "#e2e8f0", boxSizing: "border-box" }}
                    >
                      {topFeatures.map((f) => (
                        <option key={f.feature_idx} value={f.feature_idx}>
                          Feature {f.feature_idx} (weight: {f.weight.toFixed(3)})
                        </option>
                      ))}
                    </select>

                    <label style={{ display: "block", marginBottom: "8px", fontSize: 13, textAlign: "left", color: "#94a3b8" }}>
                      Alpha (strength): {featureAlpha.toFixed(2)}
                    </label>
                    <input
                      type="range"
                      min="-5"
                      max="5"
                      step="0.1"
                      value={featureAlpha}
                      onChange={(e) => setFeatureAlpha(parseFloat(e.target.value))}
                      style={{ width: "100%", marginBottom: "12px" }}
                    />

                    <button
                      onClick={applyFeaturePerturbation}
                      disabled={busy}
                      style={{
                        width: "100%",
                        padding: "10px 16px",
                        background: busy ? "#475569" : "#475569",
                        color: "white",
                        border: "none",
                        borderRadius: 6,
                        cursor: busy ? "not-allowed" : "pointer",
                        fontSize: 14,
                        fontWeight: 600,
                        transition: "all 0.2s",
                        boxShadow: busy ? "none" : "0 2px 6px rgba(0,0,0,0.3)"
                      }}
                      onMouseEnter={(e) => {
                        if (!busy) {
                          e.currentTarget.style.background = "#64748b";
                          e.currentTarget.style.transform = "translateY(-1px)";
                        }
                      }}
                      onMouseLeave={(e) => {
                        if (!busy) {
                          e.currentTarget.style.background = "#475569";
                          e.currentTarget.style.transform = "translateY(0)";
                        }
                      }}
                    >
                      {busy ? "Applying..." : "Apply Feature Perturbation"}
                    </button>
                  </div>
                )}
              </>
              )}
            </div>
          </>
        )}

          <div style={{ marginTop: "auto", padding: "16px", background: "#0f172a", borderRadius: 6, fontSize: 13, lineHeight: 1.6, border: "1px solid rgba(71, 85, 105, 0.3)", width: "100%", boxSizing: "border-box" }}>
            <strong style={{ color: "#e2e8f0" }}>What is a Neural Network?</strong>
            <p style={{ margin: "8px 0 0 0", color: "#94a3b8", textAlign: "left" }}>
              A <TermDefinition term="neural network">neural network</TermDefinition> learns patterns from data via layers of <TermDefinition term="neuron">neurons</TermDefinition> and weighted <TermDefinition term="edge">connections</TermDefinition>.
            </p>
          </div>
        </>
      )}
    </div>
  );
}
