// ModelUpload.tsx
import React, { useState } from "react";
import InfoPanel from "../UI/InfoPanel";
import TermDefinition from "../UI/TermDefinition";

type Summary = { layers: { size: number }[] };
type Props = {
  onLoadModel: (summary: Summary) => void;
  onRunComplete?: (results: any) => void;
};

export default function ModelUpload({ onLoadModel, onRunComplete }: Props) {
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
    <div style={{ width: 280, padding: 20, borderRight: "1px solid #ccc", overflowY: "auto", height: "100vh" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
        <h3 style={{ margin: 0 }}>Upload <TermDefinition term="neural network">Model</TermDefinition></h3>
        <InfoPanel
          title="How to Upload a Model"
          content={
            <div>
              <p><strong>New:</strong> You can now upload <code>.pt</code> PyTorch checkpoints. We’ll infer layer sizes and convert to the JSON structure your graph needs.</p>
              {/* keep the rest of your existing help content if useful */}
            </div>
          }
          position="bottom-right"
          size="large"
        />
      </div>

      <div style={{ marginBottom: "16px" }}>
        <label style={{ display: "block", marginBottom: "8px", fontSize: "14px", fontWeight: 500 }}>
          Select Model File (.json or .pt):
        </label>
        <input
          type="file"
          accept=".json,.pt"
          onChange={handleFile}
          disabled={busy}
          style={{ width: "100%", padding: 8, border: "1px solid #ccc", borderRadius: 4, fontSize: 14 }}
        />
      </div>

      {modelId && (
        <>
          <div style={{ padding: 12, background: "#f8f9fa", borderRadius: 4, marginBottom: 12 }}>
            <div style={{ fontWeight: 600, marginBottom: 8 }}>Weight editing</div>
            <label style={{ display: "block", marginBottom: 6, fontSize: 13 }}>Choose tensor key:</label>
            <select
              value={selectedKey ?? ""}
              onChange={(e) => setSelectedKey(e.target.value)}
              style={{ width: "100%", marginBottom: 10 }}
            >
              {weightKeys.map(k => <option key={k} value={k}>{k}</option>)}
            </select>

            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <button onClick={() => perturb("scale")} disabled={busy}>Scale</button>
              <button onClick={() => perturb("add_noise")} disabled={busy}>Add noise</button>
              <button onClick={() => perturb("set")} disabled={busy}>Set value</button>
            </div>

            <p style={{ marginTop: 10, fontSize: 12, color: "#555" }}>
              Changes are in-memory. Perturbations will affect the model run below.
            </p>
          </div>

          <div style={{ padding: 12, background: "#e8f4f8", borderRadius: 4 }}>
            <div style={{ fontWeight: 600, marginBottom: 8 }}>Run Model</div>

            <label style={{ display: "block", marginBottom: 6, fontSize: 13 }}>Environment:</label>
            <select
              value={environment}
              onChange={(e) => setEnvironment(e.target.value)}
              style={{ width: "100%", marginBottom: 10, padding: 4 }}
            >
              <option value="Walker2d-v4">Walker2d-v4</option>
              <option value="HalfCheetah-v4">HalfCheetah-v4</option>
              <option value="Hopper-v4">Hopper-v4</option>
              <option value="Ant-v4">Ant-v4</option>
              <option value="hard_stable">hard_stable (6-dim obs)</option>
            </select>

            <div style={{ marginBottom: 10 }}>
              <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13, cursor: "pointer" }}>
                <input
                  type="checkbox"
                  checked={captureActivations}
                  onChange={(e) => setCaptureActivations(e.target.checked)}
                />
                Capture activations (enables visualization)
              </label>
            </div>

            <p style={{ fontSize: 13, marginBottom: 10, color: "#555" }}>
              Test the current model (with any perturbations) in the selected environment.
            </p>
            <button
              onClick={runModel}
              disabled={isRunning}
              style={{
                width: "100%",
                padding: "8px 16px",
                background: isRunning ? "#ccc" : "#007bff",
                color: "white",
                border: "none",
                borderRadius: 4,
                cursor: isRunning ? "not-allowed" : "pointer",
                fontSize: 14,
                fontWeight: 500
              }}
            >
              {isRunning ? "Running..." : "Run Model"}
            </button>
          </div>

          {runResults && (
            <div style={{ marginTop: 12, padding: 12, background: "#d4edda", borderRadius: 4 }}>
              <div style={{ fontWeight: 600, marginBottom: 8 }}>Results</div>
              <div style={{ fontSize: 13, lineHeight: 1.6 }}>
                <div><strong>Trajectories:</strong> {runResults.num_trajectories}</div>
                <div><strong>Avg Reward:</strong> {runResults.avg_reward?.toFixed(2)}</div>
                <div><strong>Max Reward:</strong> {runResults.max_reward?.toFixed(2)}</div>
                <div><strong>Min Reward:</strong> {runResults.min_reward?.toFixed(2)}</div>
                <div><strong>Avg Length:</strong> {runResults.avg_length?.toFixed(1)} steps</div>
              </div>
            </div>
          )}

          {/* SAE Section */}
          <div style={{ marginTop: 12, padding: 12, background: "#fff3cd", borderRadius: 4 }}>
            <div style={{ fontWeight: 600, marginBottom: 8 }}>SAE Feature Analysis</div>

            {!saeLoaded ? (
              <>
                <p style={{ fontSize: 13, marginBottom: 10, color: "#856404" }}>
                  Load a Sparse Autoencoder to interpret and perturb learned features.
                </p>
                <button
                  onClick={loadSAE}
                  disabled={busy}
                  style={{
                    width: "100%",
                    padding: "8px 16px",
                    background: busy ? "#ccc" : "#ffc107",
                    color: "#000",
                    border: "none",
                    borderRadius: 4,
                    cursor: busy ? "not-allowed" : "pointer",
                    fontSize: 14,
                    fontWeight: 500
                  }}
                >
                  {busy ? "Loading..." : "Load SAE"}
                </button>
              </>
            ) : (
              <>
                <div style={{ fontSize: 12, marginBottom: 10, color: "#856404" }}>
                  <div>Latent dim: {saeInfo?.d_latent}, Top-K: {saeInfo?.k}</div>
                </div>

                {topFeatures.length > 0 && (
                  <div style={{ marginBottom: 10 }}>
                    <label style={{ display: "block", marginBottom: 6, fontSize: 13 }}>
                      Top Interpretable Features:
                    </label>
                    <select
                      value={selectedFeature ?? ""}
                      onChange={(e) => setSelectedFeature(parseInt(e.target.value))}
                      style={{ width: "100%", marginBottom: 8, fontSize: 12 }}
                    >
                      {topFeatures.map((f) => (
                        <option key={f.feature_idx} value={f.feature_idx}>
                          Feature {f.feature_idx} (weight: {f.weight.toFixed(3)})
                        </option>
                      ))}
                    </select>

                    <label style={{ display: "block", marginBottom: 6, fontSize: 13 }}>
                      Alpha (strength): {featureAlpha.toFixed(2)}
                    </label>
                    <input
                      type="range"
                      min="-5"
                      max="5"
                      step="0.1"
                      value={featureAlpha}
                      onChange={(e) => setFeatureAlpha(parseFloat(e.target.value))}
                      style={{ width: "100%", marginBottom: 10 }}
                    />

                    <button
                      onClick={applyFeaturePerturbation}
                      disabled={busy}
                      style={{
                        width: "100%",
                        padding: "8px 16px",
                        background: busy ? "#ccc" : "#ff9800",
                        color: "white",
                        border: "none",
                        borderRadius: 4,
                        cursor: busy ? "not-allowed" : "pointer",
                        fontSize: 14,
                        fontWeight: 500
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

      <div style={{ marginTop: 16, padding: 12, background: "#f8f9fa", borderRadius: 4, fontSize: 13, lineHeight: 1.6 }}>
        <strong>What is a Neural Network?</strong>
        <p style={{ margin: "8px 0 0 0" }}>
          A <TermDefinition term="neural network">neural network</TermDefinition> learns patterns from data via layers of <TermDefinition term="neuron">neurons</TermDefinition> and weighted <TermDefinition term="edge">connections</TermDefinition>.
        </p>
      </div>
    </div>
  );
}
