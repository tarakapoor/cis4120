import React from "react";
import InfoPanel from "../UI/InfoPanel";
import TermDefinition from "../UI/TermDefinition";

export default function ModelUpload({ onLoadModel }: any) {
  function handleFile(e: any) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      try {
        const json = JSON.parse(reader.result as string);
        onLoadModel(json);
      } catch (err) {
        console.error("Invalid JSON", err);
        alert("Invalid JSON file. Please make sure the file is properly formatted.");
      }
    };
    reader.readAsText(file);
  }

  return (
    <div style={{ width: 280, padding: 20, borderRight: "1px solid #ccc", overflowY: "auto", height: "100vh" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
        <h3 style={{ margin: 0 }}>Upload <TermDefinition term="neural network">Model</TermDefinition></h3>
        <InfoPanel
          title="How to Upload a Model"
          content={
            <div>
              <p style={{ marginTop: 0 }}>
                <strong>Step 1:</strong> Prepare your model file
              </p>
              <p>
                Your model should be a JSON file containing a structure with a <TermDefinition term="layer">layers</TermDefinition> array. 
                Each layer should have a <code>size</code> property indicating the number of <TermDefinition term="neuron">neurons</TermDefinition> in that layer.
              </p>
              
              <p>
                <strong>Step 2:</strong> Upload the file
              </p>
              <p>
                Click the "Choose File" button and select your JSON model file from your computer.
              </p>

              <p>
                <strong>Step 3:</strong> Visualize
              </p>
              <p>
                Once uploaded, the <TermDefinition term="neural network">neural network</TermDefinition> will appear on the right. 
                You can interact with it by clicking on <TermDefinition term="neuron">neurons</TermDefinition> to see their <TermDefinition term="connection">connections</TermDefinition>.
              </p>

              <div style={{ marginTop: "16px", padding: "12px", background: "#e8f4f8", borderRadius: "4px" }}>
                <strong>💡 Best Practice:</strong>
                <p style={{ margin: "8px 0 0 0", fontSize: "13px" }}>
                  Make sure your JSON file is valid and follows the expected format. The visualization works best with networks that have 2-10 layers.
                </p>
              </div>

              <div style={{ marginTop: "12px", padding: "12px", background: "#fff3cd", borderRadius: "4px" }}>
                <strong>📋 Example Format:</strong>
                <pre style={{ fontSize: "11px", margin: "8px 0 0 0", overflow: "auto" }}>
{`{
  "layers": [
    { "size": 4 },
    { "size": 8 },
    { "size": 3 }
  ]
}`}
                </pre>
              </div>
            </div>
          }
          position="bottom-right"
          size="large"
        />
      </div>
      
      <div style={{ marginBottom: "16px" }}>
        <label style={{ display: "block", marginBottom: "8px", fontSize: "14px", fontWeight: "500" }}>
          Select Model File (JSON):
        </label>
        <input 
          type="file" 
          accept=".json" 
          onChange={handleFile}
          style={{
            width: "100%",
            padding: "8px",
            border: "1px solid #ccc",
            borderRadius: "4px",
            fontSize: "14px"
          }}
        />
      </div>

      <div style={{ 
        padding: "12px", 
        background: "#f8f9fa", 
        borderRadius: "4px",
        fontSize: "13px",
        lineHeight: "1.6"
      }}>
        <strong>What is a Neural Network?</strong>
        <p style={{ margin: "8px 0 0 0" }}>
          A <TermDefinition term="neural network">neural network</TermDefinition> is a computing system that learns patterns from data. 
          It consists of <TermDefinition term="layer">layers</TermDefinition> of <TermDefinition term="neuron">neurons</TermDefinition> connected by <TermDefinition term="edge">edges</TermDefinition>.
        </p>
      </div>
    </div>
  );
}
