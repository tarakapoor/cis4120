import React from "react";
import TermDefinition from "./TermDefinition";

interface HelpPageProps {
  onClose: () => void;
}

export default function HelpPage({ onClose }: HelpPageProps) {
  return (
    <>
      {/* Backdrop */}
      <div
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: "rgba(0, 0, 0, 0.5)",
          zIndex: 9998
        }}
        onClick={onClose}
      />

      {/* Help Page Content */}
      <div
        style={{
          position: "fixed",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          width: "90%",
          maxWidth: "900px",
          maxHeight: "90vh",
          background: "white",
          borderRadius: "12px",
          boxShadow: "0 8px 32px rgba(0,0,0,0.2)",
          zIndex: 9999,
          overflowY: "auto",
          fontFamily: "'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          style={{
            padding: "24px 32px",
            borderBottom: "2px solid #e0e0e0",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            background: "linear-gradient(135deg, #2563eb 0%, #1e40af 100%)",
            borderRadius: "12px 12px 0 0"
          }}
        >
          <h1
            style={{
              margin: 0,
              fontSize: "28px",
              fontWeight: "700",
              color: "white",
              letterSpacing: "-0.5px"
            }}
          >
            User Guide
          </h1>
          <button
            onClick={onClose}
            style={{
              background: "rgba(255, 255, 255, 0.2)",
              border: "2px solid rgba(255, 255, 255, 0.3)",
              borderRadius: "8px",
              fontSize: "24px",
              cursor: "pointer",
              color: "white",
              padding: "4px 12px",
              lineHeight: "1",
              fontWeight: "600",
              transition: "all 0.2s"
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = "rgba(255, 255, 255, 0.3)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = "rgba(255, 255, 255, 0.2)";
            }}
            aria-label="Close help page"
          >
            ×
          </button>
        </div>

        {/* Content */}
        <div style={{ padding: "32px" }}>
          {/* Getting Started */}
          <section style={{ marginBottom: "40px" }}>
            <h2
              style={{
                fontSize: "24px",
                fontWeight: "600",
                color: "#1e293b",
                marginBottom: "16px",
                paddingBottom: "12px",
                borderBottom: "2px solid #e2e8f0"
              }}
            >
              Getting Started
            </h2>
            <div style={{ fontSize: "16px", lineHeight: "1.8", color: "#475569" }}>
              <p style={{ marginBottom: "16px" }}>
                Brainwave allows you to visualize, analyze, and manipulate neural network models. 
                The interface is designed for users familiar with neural networks who want a clean, professional tool for model inspection and experimentation.
              </p>
              <ol style={{ paddingLeft: "24px", marginBottom: "16px" }}>
                <li style={{ marginBottom: "12px" }}>
                  <strong>Upload a Model:</strong> Use the left panel to upload either a JSON model file or a PyTorch checkpoint (.pt file)
                </li>
                <li style={{ marginBottom: "12px" }}>
                  <strong>Explore the Network:</strong> The center panel displays an interactive visualization of your network structure
                </li>
                <li style={{ marginBottom: "12px" }}>
                  <strong>Adjust Weights:</strong> Click on neurons to adjust their connection weights in the right panel
                </li>
                <li style={{ marginBottom: "12px" }}>
                  <strong>Run Experiments:</strong> Test your model with different weight configurations in various environments
                </li>
              </ol>
            </div>
          </section>

          {/* Model Upload */}
          <section style={{ marginBottom: "40px" }}>
            <h2
              style={{
                fontSize: "24px",
                fontWeight: "600",
                color: "#1e293b",
                marginBottom: "16px",
                paddingBottom: "12px",
                borderBottom: "2px solid #e2e8f0"
              }}
            >
              Model Upload
            </h2>
            <div style={{ fontSize: "16px", lineHeight: "1.8", color: "#475569" }}>
              <h3 style={{ fontSize: "18px", fontWeight: "600", color: "#334155", marginTop: "20px", marginBottom: "12px" }}>
                Supported Formats
              </h3>
              <ul style={{ paddingLeft: "24px", marginBottom: "20px" }}>
                <li style={{ marginBottom: "8px" }}>
                  <strong>JSON Files:</strong> Standard JSON format containing layer information (size, structure)
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <strong>PyTorch Checkpoints (.pt):</strong> The system will automatically infer layer sizes and convert to the required format
                </li>
              </ul>

              <h3 style={{ fontSize: "18px", fontWeight: "600", color: "#334155", marginTop: "20px", marginBottom: "12px" }}>
                Weight Editing
              </h3>
              <p style={{ marginBottom: "12px" }}>
                After uploading a .pt model, you can edit weights directly:
              </p>
              <ul style={{ paddingLeft: "24px", marginBottom: "20px" }}>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Scale:</strong> Multiply all weights in a selected tensor by a factor
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Add Noise:</strong> Add Gaussian noise to weights (useful for robustness testing)
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Set Value:</strong> Set all weights in a tensor to a specific value
                </li>
              </ul>

              <h3 style={{ fontSize: "18px", fontWeight: "600", color: "#334155", marginTop: "20px", marginBottom: "12px" }}>
                Running Models
              </h3>
              <p style={{ marginBottom: "12px" }}>
                Test your model (with any perturbations) in various environments:
              </p>
              <ul style={{ paddingLeft: "24px", marginBottom: "20px" }}>
                <li style={{ marginBottom: "8px" }}>
                  Select an environment (Walker2d-v4, HalfCheetah-v4, Hopper-v4, Ant-v4, or hard_stable)
                </li>
                <li style={{ marginBottom: "8px" }}>
                  Enable "Capture Activations" to visualize activation patterns during inference
                </li>
                <li style={{ marginBottom: "8px" }}>
                  Click "Run Model" to execute and view results (rewards, trajectory lengths, etc.)
                </li>
              </ul>

              <h3 style={{ fontSize: "18px", fontWeight: "600", color: "#334155", marginTop: "20px", marginBottom: "12px" }}>
                SAE Feature Analysis
              </h3>
              <p style={{ marginBottom: "12px" }}>
                Load a Sparse Autoencoder (SAE) to interpret and perturb learned features:
              </p>
              <ul style={{ paddingLeft: "24px", marginBottom: "20px" }}>
                <li style={{ marginBottom: "8px" }}>
                  Click "Load SAE" and provide the artifacts directory path
                </li>
                <li style={{ marginBottom: "8px" }}>
                  Select interpretable features from the top-K list
                </li>
                <li style={{ marginBottom: "8px" }}>
                  Adjust the alpha (perturbation strength) and apply feature perturbations
                </li>
              </ul>
            </div>
          </section>

          {/* Network Visualization */}
          <section style={{ marginBottom: "40px" }}>
            <h2
              style={{
                fontSize: "24px",
                fontWeight: "600",
                color: "#1e293b",
                marginBottom: "16px",
                paddingBottom: "12px",
                borderBottom: "2px solid #e2e8f0"
              }}
            >
              Network Visualization
            </h2>
            <div style={{ fontSize: "16px", lineHeight: "1.8", color: "#475569" }}>
              <h3 style={{ fontSize: "18px", fontWeight: "600", color: "#334155", marginTop: "20px", marginBottom: "12px" }}>
                Understanding the Graph
              </h3>
              <ul style={{ paddingLeft: "24px", marginBottom: "20px" }}>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Nodes (Circles):</strong> Represent <TermDefinition term="neuron">neurons</TermDefinition> in the network
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Edges (Lines):</strong> Represent <TermDefinition term="edge">connections</TermDefinition> with <TermDefinition term="weight">weights</TermDefinition>
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Layer Organization:</strong> Nodes are arranged in columns representing different <TermDefinition term="layer">layers</TermDefinition>
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Edge Color:</strong> 
                  <ul style={{ marginTop: "4px", paddingLeft: "20px" }}>
                    <li><strong>Blue edges</strong> = positive weights (strengthen connections)</li>
                    <li><strong>Grey edges</strong> = negative weights (weaken or invert connections)</li>
                    <li>Darker shades indicate larger absolute weight values</li>
                  </ul>
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Edge Thickness:</strong> Thicker edges indicate larger weight magnitude (stronger absolute values)
                </li>
              </ul>

              <h3 style={{ fontSize: "18px", fontWeight: "600", color: "#334155", marginTop: "20px", marginBottom: "12px" }}>
                Interactions
              </h3>
              <ul style={{ paddingLeft: "24px", marginBottom: "20px" }}>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Click a Neuron:</strong> Select it to view and adjust its outgoing weights
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Click Background:</strong> Deselect and return to the overview state
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Scroll to Zoom:</strong> Use mouse wheel to zoom in/out
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Drag to Pan:</strong> Click and drag to move around the visualization
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Hover:</strong> Hover over neurons to see their IDs
                </li>
              </ul>

              <h3 style={{ fontSize: "18px", fontWeight: "600", color: "#334155", marginTop: "20px", marginBottom: "12px" }}>
                Labels
              </h3>
              <ul style={{ paddingLeft: "24px", marginBottom: "20px" }}>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Layer Labels:</strong> Toggle to show/hide layer names (Input, Hidden, Output) and feature counts
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Node Labels:</strong> Toggle to show/hide custom labels for input and output nodes
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Edit Labels:</strong> Click on any input/output node label to customize it (e.g., "Velocity", "Position")
                </li>
              </ul>
            </div>
          </section>

          {/* Weight Adjustment */}
          <section style={{ marginBottom: "40px" }}>
            <h2
              style={{
                fontSize: "24px",
                fontWeight: "600",
                color: "#1e293b",
                marginBottom: "16px",
                paddingBottom: "12px",
                borderBottom: "2px solid #e2e8f0"
              }}
            >
              Weight Adjustment
            </h2>
            <div style={{ fontSize: "16px", lineHeight: "1.8", color: "#475569" }}>
              <p style={{ marginBottom: "16px" }}>
                When you select a neuron, the right panel displays all its outgoing connections. You can adjust these weights to <TermDefinition term="steering">steer</TermDefinition> the <TermDefinition term="policy">policy</TermDefinition> behavior.
              </p>

              <h3 style={{ fontSize: "18px", fontWeight: "600", color: "#334155", marginTop: "20px", marginBottom: "12px" }}>
                Using the Sliders
              </h3>
              <ul style={{ paddingLeft: "24px", marginBottom: "20px" }}>
                <li style={{ marginBottom: "8px" }}>
                  Each connection has a slider ranging from -1.0 to 1.0
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Positive Values:</strong> Strengthen the connection (signal enhancement)
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Negative Values:</strong> Weaken or invert the connection (signal suppression)
                </li>
                <li style={{ marginBottom: "8px" }}>
                  Changes are reflected in real-time in the network visualization
                </li>
              </ul>

              <h3 style={{ fontSize: "18px", fontWeight: "600", color: "#334155", marginTop: "20px", marginBottom: "12px" }}>
                Before/After Comparison
              </h3>
              <ul style={{ paddingLeft: "24px", marginBottom: "20px" }}>
                <li style={{ marginBottom: "8px" }}>
                  The panel shows both original ("Before") and current ("After") weight values
                </li>
                <li style={{ marginBottom: "8px" }}>
                  Modified weights are highlighted to make changes easy to track
                </li>
                <li style={{ marginBottom: "8px" }}>
                  Use "Reset This Weight" to restore a single connection, or "Reset All" to restore all connections
                </li>
              </ul>

              <h3 style={{ fontSize: "18px", fontWeight: "600", color: "#334155", marginTop: "20px", marginBottom: "12px" }}>
                Best Practices
              </h3>
              <ul style={{ paddingLeft: "24px", marginBottom: "20px" }}>
                <li style={{ marginBottom: "8px" }}>
                  Start with small perturbations to observe gradual effects
                </li>
                <li style={{ marginBottom: "8px" }}>
                  Watch how edge thickness and opacity change as you adjust weights
                </li>
                <li style={{ marginBottom: "8px" }}>
                  Experiment with different neurons to understand their roles in the network
                </li>
                <li style={{ marginBottom: "8px" }}>
                  Use the reset functions to return to baseline when needed
                </li>
              </ul>
            </div>
          </section>

          {/* Activation Viewer */}
          <section style={{ marginBottom: "40px" }}>
            <h2
              style={{
                fontSize: "24px",
                fontWeight: "600",
                color: "#1e293b",
                marginBottom: "16px",
                paddingBottom: "12px",
                borderBottom: "2px solid #e2e8f0"
              }}
            >
              Activation Viewer
            </h2>
            <div style={{ fontSize: "16px", lineHeight: "1.8", color: "#475569" }}>
              <p style={{ marginBottom: "16px" }}>
                When you run a model with "Capture Activations" enabled, the Activation Viewer appears at the bottom of the screen. 
                This tool allows you to step through the inference process and observe how activations flow through the network.
              </p>

              <h3 style={{ fontSize: "18px", fontWeight: "600", color: "#334155", marginTop: "20px", marginBottom: "12px" }}>
                Playback Controls
              </h3>
              <ul style={{ paddingLeft: "24px", marginBottom: "20px" }}>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Play/Pause:</strong> Automatically step through timesteps
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Step Forward/Backward:</strong> Move one timestep at a time
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Reset:</strong> Return to the first timestep
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Speed Control:</strong> Adjust playback speed (0.5x, 1x, 2x, 5x)
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Timeline Scrubber:</strong> Drag to jump to any timestep
                </li>
              </ul>

              <h3 style={{ fontSize: "18px", fontWeight: "600", color: "#334155", marginTop: "20px", marginBottom: "12px" }}>
                Data Display
              </h3>
              <ul style={{ paddingLeft: "24px", marginBottom: "20px" }}>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Observation:</strong> Current input state to the network
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Action:</strong> Network's output/prediction
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <strong>Layer Activations:</strong> Visual representation of activation values for each layer
                </li>
                <li style={{ marginBottom: "8px" }}>
                  Activation intensity is color-coded: blue for positive, darker shades for stronger activations
                </li>
              </ul>
            </div>
          </section>

          {/* Terminology */}
          <section style={{ marginBottom: "40px" }}>
            <h2
              style={{
                fontSize: "24px",
                fontWeight: "600",
                color: "#1e293b",
                marginBottom: "16px",
                paddingBottom: "12px",
                borderBottom: "2px solid #e2e8f0"
              }}
            >
              Terminology
            </h2>
            <div style={{ fontSize: "16px", lineHeight: "1.8", color: "#475569" }}>
              <p style={{ marginBottom: "16px" }}>
                Throughout the interface, key terms are underlined and clickable. Click any underlined term to see its definition. 
                Common terms include:
              </p>
              <ul style={{ paddingLeft: "24px", marginBottom: "20px" }}>
                <li style={{ marginBottom: "8px" }}>
                  <TermDefinition term="neural network">Neural Network</TermDefinition> - A computational model inspired by biological neural networks
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <TermDefinition term="neuron">Neuron</TermDefinition> - A basic processing unit in a neural network
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <TermDefinition term="layer">Layer</TermDefinition> - A collection of neurons that process information together
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <TermDefinition term="weight">Weight</TermDefinition> - A parameter that determines the strength of a connection
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <TermDefinition term="edge">Edge</TermDefinition> - A connection between two neurons
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <TermDefinition term="activation">Activation</TermDefinition> - The output value of a neuron after processing
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <TermDefinition term="policy">Policy</TermDefinition> - The strategy or function that maps observations to actions
                </li>
                <li style={{ marginBottom: "8px" }}>
                  <TermDefinition term="steering">Steering</TermDefinition> - The process of adjusting weights to change model behavior
                </li>
              </ul>
            </div>
          </section>

          {/* Tips & Tricks */}
          <section>
            <h2
              style={{
                fontSize: "24px",
                fontWeight: "600",
                color: "#1e293b",
                marginBottom: "16px",
                paddingBottom: "12px",
                borderBottom: "2px solid #e2e8f0"
              }}
            >
              Tips & Tricks
            </h2>
            <div style={{ fontSize: "16px", lineHeight: "1.8", color: "#475569" }}>
              <ul style={{ paddingLeft: "24px", marginBottom: "20px" }}>
                <li style={{ marginBottom: "12px" }}>
                  <strong>Use the Info Panels:</strong> Click the "?" buttons throughout the interface for context-specific help
                </li>
                <li style={{ marginBottom: "12px" }}>
                  <strong>Keyboard Shortcuts:</strong> Press Enter to save when editing node labels, Escape to cancel
                </li>
                <li style={{ marginBottom: "12px" }}>
                  <strong>Visual Feedback:</strong> Pay attention to how edge thickness and opacity change - they provide immediate feedback on weight adjustments
                </li>
                <li style={{ marginBottom: "12px" }}>
                  <strong>Experimentation:</strong> Don't hesitate to try different weight configurations - you can always reset to original values
                </li>
                <li style={{ marginBottom: "12px" }}>
                  <strong>Large Networks:</strong> Use zoom and pan to navigate large networks effectively
                </li>
                <li style={{ marginBottom: "12px" }}>
                  <strong>SAE Integration:</strong> For advanced analysis, load SAE models to interpret learned features and apply targeted perturbations
                </li>
              </ul>
            </div>
          </section>
        </div>
      </div>
    </>
  );
}

