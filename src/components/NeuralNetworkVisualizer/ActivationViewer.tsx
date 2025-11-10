import React, { useState, useEffect, useRef } from "react";

type ActivationData = {
  timestep: number;
  observation: number[];
  action: number[];
  [key: string]: any; // layer_0, layer_2, etc.
};

type Props = {
  activations: ActivationData[];
  model: { layers: { size: number }[] };
  onTimestepChange?: (timestep: number, activations: ActivationData) => void;
};

export default function ActivationViewer({ activations, model, onTimestepChange }: Props) {
  const [currentTimestep, setCurrentTimestep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const intervalRef = useRef<number | null>(null);

  const maxTimestep = activations.length - 1;

  // Play/pause effect
  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = window.setInterval(() => {
        setCurrentTimestep((prev) => {
          if (prev >= maxTimestep) {
            setIsPlaying(false);
            return maxTimestep;
          }
          return prev + 1;
        });
      }, 1000 / playbackSpeed);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isPlaying, playbackSpeed, maxTimestep]);

  // Notify parent of timestep changes
  useEffect(() => {
    if (onTimestepChange && activations[currentTimestep]) {
      onTimestepChange(currentTimestep, activations[currentTimestep]);
    }
  }, [currentTimestep, activations, onTimestepChange]);

  const handleScrub = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newTimestep = parseInt(e.target.value);
    setCurrentTimestep(newTimestep);
    setIsPlaying(false);
  };

  const togglePlay = () => {
    if (currentTimestep >= maxTimestep) {
      setCurrentTimestep(0);
    }
    setIsPlaying(!isPlaying);
  };

  const stepForward = () => {
    setCurrentTimestep(Math.min(currentTimestep + 1, maxTimestep));
    setIsPlaying(false);
  };

  const stepBackward = () => {
    setCurrentTimestep(Math.max(currentTimestep - 1, 0));
    setIsPlaying(false);
  };

  const reset = () => {
    setCurrentTimestep(0);
    setIsPlaying(false);
  };

  if (!activations || activations.length === 0) {
    return (
      <div style={{ padding: 20, textAlign: "center", color: "#666" }}>
        No activation data available. Run the model with "Capture Activations" enabled.
      </div>
    );
  }

  const currentData = activations[currentTimestep];

  return (
    <div style={{ padding: 20, borderTop: "1px solid #ddd" }}>
      <h3 style={{ margin: "0 0 16px 0", fontSize: 18 }}>Activation Viewer</h3>

      {/* Controls */}
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: 12,
        marginBottom: 16,
        padding: 12,
        background: "#f8f9fa",
        borderRadius: 4
      }}>
        <button
          onClick={reset}
          style={{
            padding: "6px 12px",
            background: "#6c757d",
            color: "white",
            border: "none",
            borderRadius: 4,
            cursor: "pointer",
            fontSize: 14
          }}
        >
          ⏮ Reset
        </button>

        <button
          onClick={stepBackward}
          disabled={currentTimestep === 0}
          style={{
            padding: "6px 12px",
            background: currentTimestep === 0 ? "#ccc" : "#007bff",
            color: "white",
            border: "none",
            borderRadius: 4,
            cursor: currentTimestep === 0 ? "not-allowed" : "pointer",
            fontSize: 14
          }}
        >
          ◀ Step
        </button>

        <button
          onClick={togglePlay}
          style={{
            padding: "8px 16px",
            background: "#28a745",
            color: "white",
            border: "none",
            borderRadius: 4,
            cursor: "pointer",
            fontSize: 16,
            fontWeight: 600
          }}
        >
          {isPlaying ? "⏸ Pause" : "▶ Play"}
        </button>

        <button
          onClick={stepForward}
          disabled={currentTimestep === maxTimestep}
          style={{
            padding: "6px 12px",
            background: currentTimestep === maxTimestep ? "#ccc" : "#007bff",
            color: "white",
            border: "none",
            borderRadius: 4,
            cursor: currentTimestep === maxTimestep ? "not-allowed" : "pointer",
            fontSize: 14
          }}
        >
          Step ▶
        </button>

        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          <label style={{ fontSize: 13, fontWeight: 500 }}>Speed:</label>
          <select
            value={playbackSpeed}
            onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
            style={{ padding: "4px 8px", borderRadius: 4, border: "1px solid #ccc" }}
          >
            <option value={0.5}>0.5x</option>
            <option value={1}>1x</option>
            <option value={2}>2x</option>
            <option value={5}>5x</option>
          </select>
        </div>
      </div>

      {/* Timeline Scrubber */}
      <div style={{ marginBottom: 20 }}>
        <div style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 8
        }}>
          <span style={{ fontSize: 14, fontWeight: 500 }}>
            Timestep: {currentTimestep} / {maxTimestep}
          </span>
          <span style={{ fontSize: 13, color: "#666" }}>
            {activations.length} frames captured
          </span>
        </div>

        <input
          type="range"
          min={0}
          max={maxTimestep}
          value={currentTimestep}
          onChange={handleScrub}
          style={{
            width: "100%",
            height: 8,
            borderRadius: 4,
            appearance: "none",
            background: `linear-gradient(to right, #007bff 0%, #007bff ${(currentTimestep / maxTimestep) * 100}%, #ddd ${(currentTimestep / maxTimestep) * 100}%, #ddd 100%)`
          }}
        />
      </div>

      {/* Current Data Display */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: 16
      }}>
        <div style={{
          padding: 12,
          background: "#e8f4f8",
          borderRadius: 4,
          fontSize: 13
        }}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Observation</div>
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(60px, 1fr))",
            gap: 4
          }}>
            {currentData.observation.map((val, idx) => (
              <div
                key={idx}
                style={{
                  padding: 4,
                  background: "white",
                  borderRadius: 2,
                  textAlign: "center",
                  fontSize: 11
                }}
              >
                {val.toFixed(3)}
              </div>
            ))}
          </div>
        </div>

        <div style={{
          padding: 12,
          background: "#d4edda",
          borderRadius: 4,
          fontSize: 13
        }}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Action</div>
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(60px, 1fr))",
            gap: 4
          }}>
            {currentData.action.map((val, idx) => (
              <div
                key={idx}
                style={{
                  padding: 4,
                  background: "white",
                  borderRadius: 2,
                  textAlign: "center",
                  fontSize: 11
                }}
              >
                {val.toFixed(3)}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Layer Activations */}
      <div style={{ marginTop: 16 }}>
        <div style={{ fontWeight: 600, marginBottom: 8 }}>Layer Activations</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {Object.keys(currentData)
            .filter(key => key.startsWith('layer_'))
            .sort()
            .map(layerKey => {
              const layerData = currentData[layerKey][0]; // First element of batch
              const layerIdx = parseInt(layerKey.split('_')[1]);

              return (
                <div
                  key={layerKey}
                  style={{
                    padding: 10,
                    background: "#fff3cd",
                    borderRadius: 4,
                    fontSize: 12
                  }}
                >
                  <div style={{ fontWeight: 600, marginBottom: 6 }}>
                    {layerKey.replace('_', ' ')} ({layerData.length} neurons)
                  </div>
                  <div style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fill, minmax(40px, 1fr))",
                    gap: 3,
                    maxHeight: 100,
                    overflowY: "auto"
                  }}>
                    {layerData.map((val: number, idx: number) => {
                      const intensity = Math.min(Math.abs(val), 1);
                      const color = val >= 0
                        ? `rgba(74, 144, 226, ${intensity})`
                        : `rgba(255, 65, 54, ${intensity})`;

                      return (
                        <div
                          key={idx}
                          title={`Neuron ${idx}: ${val.toFixed(4)}`}
                          style={{
                            padding: 3,
                            background: color,
                            borderRadius: 2,
                            textAlign: "center",
                            fontSize: 9,
                            color: intensity > 0.5 ? "white" : "black",
                            cursor: "help"
                          }}
                        >
                          {idx}
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
        </div>
      </div>
    </div>
  );
}
