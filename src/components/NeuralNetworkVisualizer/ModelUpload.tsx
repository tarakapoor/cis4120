import React from "react";

export default function ModelUpload({ onLoadModel }) {
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
      }
    };
    reader.readAsText(file);
  }

  return (
    <div style={{ width: 240, padding: 20, borderRight: "1px solid #ccc" }}>
      <h3>Upload Model</h3>
      <input type="file" accept=".json" onChange={handleFile} />
    </div>
  );
}
