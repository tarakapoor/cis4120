import React, { useState } from "react";
import ModelUpload from "./components/NeuralNetworkVisualizer/ModelUpload";
import NetworkGraph from "./components/NeuralNetworkVisualizer/NetworkGraph";
import ActivationViewer from "./components/NeuralNetworkVisualizer/ActivationViewer";
import InfoPanel from "./components/UI/InfoPanel";
import TermDefinition from "./components/UI/TermDefinition";
import HelpPage from "./components/UI/HelpPage";
import DynamicBackground from "./components/UI/DynamicBackground";


export default function App() {
  const [model, setModel] = useState(null);
  const [activations, setActivations] = useState(null);
  const [currentActivation, setCurrentActivation] = useState(null);
  const [showHelp, setShowHelp] = useState(false);

  return (
    <div style={{ display: "flex", height: "100vh", flexDirection: "column", fontFamily: "'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", position: "relative" }}>
      <DynamicBackground />
      <div style={{ 
        padding: "16px 24px", 
        background: "linear-gradient(135deg, #1e40af 0%, #2563eb 100%)", 
        borderBottom: "2px solid #1e3a8a",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        boxShadow: "0 2px 12px rgba(0,0,0,0.15)",
        position: "relative",
        zIndex: 10
      }}>
        <h1 style={{ margin: 0, fontSize: "24px", color: "white", fontWeight: "600", letterSpacing: "-0.5px" }}>
          Neural Network Debugger
        </h1>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <button
            onClick={() => setShowHelp(true)}
            style={{
              padding: "8px 16px",
              background: "rgba(255, 255, 255, 0.15)",
              border: "2px solid rgba(255, 255, 255, 0.3)",
              borderRadius: "8px",
              color: "white",
              fontSize: "14px",
              fontWeight: "500",
              cursor: "pointer",
              transition: "all 0.2s",
              fontFamily: "'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = "rgba(255, 255, 255, 0.25)";
              e.currentTarget.style.borderColor = "rgba(255, 255, 255, 0.5)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = "rgba(255, 255, 255, 0.15)";
              e.currentTarget.style.borderColor = "rgba(255, 255, 255, 0.3)";
            }}
          >
            Help
          </button>
          <InfoPanel
          title="Welcome to the Neural Network Debugger"
          content={
            <div style={{ fontFamily: "'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" }}>
              <p style={{ marginTop: 0, lineHeight: "1.6", color: "#2c3e50" }}>
                This tool helps you visualize and understand <TermDefinition term="neural network">neural networks</TermDefinition> 
                without needing deep technical knowledge.
              </p>
              
              <p style={{ fontWeight: "600", marginBottom: "8px", color: "#2c3e50" }}>
                Getting Started:
              </p>
              <ol style={{ paddingLeft: "20px", margin: "8px 0", lineHeight: "1.6", color: "#34495e" }}>
                <li>Upload a model JSON file using the panel on the left</li>
                <li>Explore the network visualization by clicking on <TermDefinition term="neuron">neurons</TermDefinition></li>
                <li>Click on any underlined term to learn what it means</li>
                <li>Use the <strong>?</strong> buttons to get detailed help</li>
              </ol>

              <div style={{ marginTop: "16px", padding: "16px", background: "#eff6ff", borderRadius: "8px", borderLeft: "4px solid #2563eb" }}>
                <strong style={{ color: "#1e40af" }}>💡 Tips:</strong>
                <ul style={{ margin: "8px 0 0 0", paddingLeft: "20px", fontSize: "14px", lineHeight: "1.6", color: "#475569" }}>
                  <li>Click underlined terms to see definitions</li>
                  <li>Start by exploring different <TermDefinition term="layer">layers</TermDefinition> of the network</li>
                  <li>Try clicking different <TermDefinition term="neuron">neurons</TermDefinition> to see how they connect</li>
                  <li>Use zoom and pan to get a better view of large networks</li>
                </ul>
              </div>

              <div style={{ marginTop: "12px", padding: "16px", background: "#f8fafc", borderRadius: "8px", borderLeft: "4px solid #64748b" }}>
                <strong style={{ color: "#475569" }}>🎓 Key Concepts:</strong>
                <p style={{ margin: "8px 0 0 0", fontSize: "14px", lineHeight: "1.6", color: "#64748b" }}>
                  A <TermDefinition term="neural network">neural network</TermDefinition> is a computational model that learns from examples. 
                  It's made up of <TermDefinition term="neuron">neurons</TermDefinition> connected by <TermDefinition term="edge">edges</TermDefinition> with <TermDefinition term="weight">weights</TermDefinition>. 
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
      </div>

      {showHelp && <HelpPage onClose={() => setShowHelp(false)} />}
      
      <div style={{ display: "flex", flex: 1, overflow: "hidden", position: "relative", zIndex: 1 }}>
        <ModelUpload
          onLoadModel={(data) => setModel(data)}
          onRunComplete={(results) => {
            if (results && results.activations) {
              setActivations(results.activations);
            }
          }}
        />

        <div style={{ flex: 1, overflow: "auto", display: "flex", flexDirection: "column", position: "relative", zIndex: 1 }}>
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
              color: "#64748b",
              height: "100%",
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
              alignItems: "center",
              background: "rgba(248, 250, 252, 0.7)",
              backdropFilter: "blur(10px)",
              position: "relative",
              zIndex: 1
            }}>
              <div style={{ maxWidth: "600px" }}>
                <h2 style={{ color: "#1e293b", marginBottom: "16px", fontSize: "28px", fontWeight: "600", fontFamily: "'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" }}>
                  Welcome to the Neural Network Visualizer
                </h2>
                <p style={{ fontSize: "16px", lineHeight: "1.6", marginBottom: "24px", color: "#475569", fontFamily: "'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" }}>
                  Upload a <TermDefinition term="neural network">neural network</TermDefinition> model file to get started. 
                  The model should be a JSON file or PyTorch checkpoint (.pt) containing information about the network's <TermDefinition term="layer">layers</TermDefinition> and structure.
                </p>
                <div style={{ 
                  padding: "24px", 
                  background: "white", 
                  borderRadius: "12px",
                  textAlign: "left",
                  marginTop: "20px",
                  boxShadow: "0 2px 12px rgba(0,0,0,0.08)",
                  border: "1px solid #e2e8f0"
                }}>
                  <strong style={{ fontSize: "16px", color: "#1e293b", fontFamily: "'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" }}>What you can do:</strong>
                  <ul style={{ marginTop: "12px", lineHeight: "1.8", color: "#475569", fontFamily: "'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" }}>
                    <li>Visualize the structure of your <TermDefinition term="neural network">neural network</TermDefinition></li>
                    <li>Explore connections between <TermDefinition term="neuron">neurons</TermDefinition></li>
                    <li>Adjust weights to steer policy behavior</li>
                    <li>Run models in various environments and view activations</li>
                    <li>Learn about key concepts through interactive definitions</li>
                  </ul>
                </div>
                <p style={{ marginTop: "24px", fontSize: "14px", color: "#94a3b8", fontFamily: "'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" }}>
                  Click the <strong>Help</strong> button in the top right for a comprehensive guide, or use the info panels throughout the interface.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// export default function App() {
//   const [model, setModel] = useState(null);
//   const [activations, setActivations] = useState(null);
//   const [currentActivation, setCurrentActivation] = useState(null);

//   return (
//     <div style={{ display: "flex", height: "100vh", flexDirection: "column" }}>
//       <div style={{ 
//         padding: "12px 20px", 
//         background: "#f8f9fa", 
//         borderBottom: "1px solid #ddd",
//         display: "flex",
//         justifyContent: "space-between",
//         alignItems: "center"
//       }}>
//         <h1 style={{ margin: 0, fontSize: "20px", color: "#333" }}>
//           Neural Network Debugger
//         </h1>
//         <InfoPanel
//           title="Welcome to the Neural Network Debugger"
//           content={
//             <div>
//               <p style={{ marginTop: 0 }}>
//                 This tool helps you visualize and understand <TermDefinition term="neural network">neural networks</TermDefinition> 
//                 without needing deep technical knowledge.
//               </p>
              
//               <p>
//                 <strong>Getting Started:</strong>
//               </p>
//               <ol style={{ paddingLeft: "20px", margin: "8px 0" }}>
//                 <li>Upload a model JSON file using the panel on the left</li>
//                 <li>Explore the network visualization by clicking on <TermDefinition term="neuron">neurons</TermDefinition></li>
//                 <li>Click on any underlined term to learn what it means</li>
//                 <li>Use the <strong>?</strong> buttons to get detailed help</li>
//               </ol>

//               <div style={{ marginTop: "16px", padding: "12px", background: "#e8f4f8", borderRadius: "4px" }}>
//                 <strong>💡 Tips for Beginners:</strong>
//                 <ul style={{ margin: "8px 0 0 0", paddingLeft: "20px", fontSize: "13px" }}>
//                   <li>Don't worry if terms seem confusing - click them to learn!</li>
//                   <li>Start by exploring different <TermDefinition term="layer">layers</TermDefinition> of the network</li>
//                   <li>Try clicking different <TermDefinition term="neuron">neurons</TermDefinition> to see how they connect</li>
//                   <li>Use zoom and pan to get a better view of large networks</li>
//                 </ul>
//               </div>

//               <div style={{ marginTop: "12px", padding: "12px", background: "#d4edda", borderRadius: "4px" }}>
//                 <strong>🎓 Key Concepts:</strong>
//                 <p style={{ margin: "8px 0 0 0", fontSize: "13px" }}>
//                   A <TermDefinition term="neural network">neural network</TermDefinition> is like a digital brain that learns from examples. 
//                   It's made up of <TermDefinition term="neuron">neurons</TermDefinition> (brain cells) connected by <TermDefinition term="edge">edges</TermDefinition> (pathways). 
//                   Information flows from the <TermDefinition term="input">input</TermDefinition> through <TermDefinition term="hidden layer">hidden layers</TermDefinition> 
//                   to the <TermDefinition term="output">output</TermDefinition>, where the network makes its prediction or decision.
//                 </p>
//               </div>
//             </div>
//           }
//           position="top-right"
//           size="large"
//         />
//       </div>
      
//       <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
//         <ModelUpload
//           onLoadModel={(data) => setModel(data)}
//           onRunComplete={(results) => {
//             if (results && results.activations) {
//               setActivations(results.activations);
//             }
//           }}
//         />

//         <div style={{ flex: 1, overflow: "auto", display: "flex", flexDirection: "column" }}>
//           {model ? (
//             <>
//               <NetworkGraph model={model} currentActivation={currentActivation} />
//               {activations && (
//                 <ActivationViewer
//                   activations={activations}
//                   model={model}
//                   onTimestepChange={(timestep, activation) => setCurrentActivation(activation)}
//                 />
//               )}
//             </>
//           ) : (
//             <div style={{ 
//               padding: "40px 20px", 
//               textAlign: "center",
//               color: "#666",
//               height: "100%",
//               display: "flex",
//               flexDirection: "column",
//               justifyContent: "center",
//               alignItems: "center"
//             }}>
//               <div style={{ maxWidth: "600px" }}>
//                 <h2 style={{ color: "#333", marginBottom: "16px" }}>
//                   Welcome to the Neural Network Visualizer
//                 </h2>
//                 <p style={{ fontSize: "16px", lineHeight: "1.6", marginBottom: "24px" }}>
//                   Upload a <TermDefinition term="neural network">neural network</TermDefinition> model file to get started. 
//                   The model should be a JSON file containing information about the network's <TermDefinition term="layer">layers</TermDefinition> and structure.
//                 </p>
//                 <div style={{ 
//                   padding: "20px", 
//                   background: "#f8f9fa", 
//                   borderRadius: "8px",
//                   textAlign: "left",
//                   marginTop: "20px"
//                 }}>
//                   <strong>What you can do:</strong>
//                   <ul style={{ marginTop: "12px", lineHeight: "1.8" }}>
//                     <li>Visualize the structure of your <TermDefinition term="neural network">neural network</TermDefinition></li>
//                     <li>Explore connections between <TermDefinition term="neuron">neurons</TermDefinition></li>
//                     <li>Learn about key concepts through interactive definitions</li>
//                     <li>Understand how information flows through the network</li>
//                   </ul>
//                 </div>
//                 <p style={{ marginTop: "24px", fontSize: "14px", color: "#888" }}>
//                   Click the <strong>?</strong> button in the top right for detailed help, or use the info panel in the upload section.
//                 </p>
//               </div>
//             </div>
//           )}
//         </div>
//       </div>
//     </div>
//   );
// }
