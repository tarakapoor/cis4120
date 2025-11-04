import React, { useState } from "react";
import ModelUpload from "./components/NeuralNetworkVisualizer/ModelUpload";
import NetworkGraph from "./components/NeuralNetworkVisualizer/NetworkGraph";

export default function App() {
  const [model, setModel] = useState(null);

  return (
    <div style={{ display: "flex", height: "100vh" }}>
      <ModelUpload onLoadModel={(data) => setModel(data)} />

      <div style={{ flex: 1 }}>
        {model ? (
          <NetworkGraph model={model} />
        ) : (
          <div style={{ padding: 20 }}>Upload a model JSON to visualize.</div>
        )}
      </div>
    </div>
  );
}
