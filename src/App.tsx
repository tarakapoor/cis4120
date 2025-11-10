import React, { useState } from "react";
import ModelUpload from "./components/NeuralNetworkVisualizer/ModelUpload";
import NetworkGraph from "./components/NeuralNetworkVisualizer/NetworkGraph";
import ActivationViewer from "./components/NeuralNetworkVisualizer/ActivationViewer";
import InfoPanel from "./components/UI/InfoPanel";
import TermDefinition from "./components/UI/TermDefinition";

export default function App() {
  const [model, setModel] = useState(null);
  const [activations, setActivations] = useState(null);
  const [currentActivation, setCurrentActivation] = useState(null);

  return (
    <div style={{ display: "flex", height: "100vh", flexDirection: "column" }}>
      <div style={{ 
        padding: "12px 20px", 
        background: "#f8f9fa", 
        borderBottom: "1px solid #ddd",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center"
      }}>
        <h1 style={{ margin: 0, fontSize: "20px", color: "#333" }}>
          Neural Network Debugger
        </h1>
        <InfoPanel
          title="Welcome to the Neural Network Debugger"
          content={
            <div>
              <p style={{ marginTop: 0 }}>
                This tool helps you visualize and understand <TermDefinition term="neural network">neural networks</TermDefinition> 
                without needing deep technical knowledge.
              </p>
              
              <p>
                <strong>Getting Started:</strong>
              </p>
              <ol style={{ paddingLeft: "20px", margin: "8px 0" }}>
                <li>Upload a model JSON file using the panel on the left</li>
                <li>Explore the network visualization by clicking on <TermDefinition term="neuron">neurons</TermDefinition></li>
                <li>Click on any underlined term to learn what it means</li>
                <li>Use the <strong>?</strong> buttons to get detailed help</li>
              </ol>

              <div style={{ marginTop: "16px", padding: "12px", background: "#e8f4f8", borderRadius: "4px" }}>
                <strong>💡 Tips for Beginners:</strong>
                <ul style={{ margin: "8px 0 0 0", paddingLeft: "20px", fontSize: "13px" }}>
                  <li>Don't worry if terms seem confusing - click them to learn!</li>
                  <li>Start by exploring different <TermDefinition term="layer">layers</TermDefinition> of the network</li>
                  <li>Try clicking different <TermDefinition term="neuron">neurons</TermDefinition> to see how they connect</li>
                  <li>Use zoom and pan to get a better view of large networks</li>
                </ul>
              </div>

              <div style={{ marginTop: "12px", padding: "12px", background: "#d4edda", borderRadius: "4px" }}>
                <strong>🎓 Key Concepts:</strong>
                <p style={{ margin: "8px 0 0 0", fontSize: "13px" }}>
                  A <TermDefinition term="neural network">neural network</TermDefinition> is like a digital brain that learns from examples. 
                  It's made up of <TermDefinition term="neuron">neurons</TermDefinition> (brain cells) connected by <TermDefinition term="edge">edges</TermDefinition> (pathways). 
                  Information flows from the <TermDefinition term="input">input</TermDefinition> through <TermDefinition term="hidden layer">hidden layers</TermDefinition> 
                  to the <TermDefinition term="output">output</TermDefinition>, where the network makes its prediction or decision.
                </p>
              </div>
            </div>
          }
          position="top-right"
          size="large"
        />
      </div>
      
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        <ModelUpload
          onLoadModel={(data) => setModel(data)}
          onRunComplete={(results) => {
            if (results && results.activations) {
              setActivations(results.activations);
            }
          }}
        />

        <div style={{ flex: 1, overflow: "auto", display: "flex", flexDirection: "column" }}>
          {model ? (
            <>
              <NetworkGraph model={model} currentActivation={currentActivation} />
              {activations && (
                <ActivationViewer
                  activations={activations}
                  model={model}
                  onTimestepChange={(timestep, activation) => setCurrentActivation(activation)}
                />
              )}
            </>
          ) : (
            <div style={{ 
              padding: "40px 20px", 
              textAlign: "center",
              color: "#666",
              height: "100%",
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
              alignItems: "center"
            }}>
              <div style={{ maxWidth: "600px" }}>
                <h2 style={{ color: "#333", marginBottom: "16px" }}>
                  Welcome to the Neural Network Visualizer
                </h2>
                <p style={{ fontSize: "16px", lineHeight: "1.6", marginBottom: "24px" }}>
                  Upload a <TermDefinition term="neural network">neural network</TermDefinition> model file to get started. 
                  The model should be a JSON file containing information about the network's <TermDefinition term="layer">layers</TermDefinition> and structure.
                </p>
                <div style={{ 
                  padding: "20px", 
                  background: "#f8f9fa", 
                  borderRadius: "8px",
                  textAlign: "left",
                  marginTop: "20px"
                }}>
                  <strong>What you can do:</strong>
                  <ul style={{ marginTop: "12px", lineHeight: "1.8" }}>
                    <li>Visualize the structure of your <TermDefinition term="neural network">neural network</TermDefinition></li>
                    <li>Explore connections between <TermDefinition term="neuron">neurons</TermDefinition></li>
                    <li>Learn about key concepts through interactive definitions</li>
                    <li>Understand how information flows through the network</li>
                  </ul>
                </div>
                <p style={{ marginTop: "24px", fontSize: "14px", color: "#888" }}>
                  Click the <strong>?</strong> button in the top right for detailed help, or use the info panel in the upload section.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
