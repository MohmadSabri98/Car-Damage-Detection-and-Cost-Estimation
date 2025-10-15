import { useState } from "react";
import "../../App.css";
import DamageCard from "./DamageCard";
import SummarySection from "./SummarySection";

export default function FileUploader() {
  const [responseData, setResponseData] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);

  // ✅ Helper function to clean text
  const formatText = (text) => {
    if (text === null || text === undefined) return "";
    return String(text).replace(/[^a-zA-Z0-9\u0600-\u06FF]+/g, " ").trim();
  };

  const handleFile = async (file) => {
    if (!file) return;

    const form = new FormData();
    form.append("file", file);

    setUploading(true);
    setProgress(0);

    try {
      // simulate upload progress
      const interval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) {
            clearInterval(interval);
            return prev;
          }
          return prev + 10;
        });
      }, 300);

      const res = await fetch("http://localhost:8000/infer", {
        method: "POST",
        body: form,
        headers: {
          Accept: "application/json",
        },
      });

      const data = await res.json();
      setResponseData(data);

      setProgress(100);
      setTimeout(() => setUploading(false), 500);
    } catch (err) {
      console.error("Upload failed:", err);
      setUploading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    handleFile(e.dataTransfer.files[0]);
  };

  return (
    <div className="uploader-container">
      <div
        className="dropzone"
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        onClick={() => document.getElementById("fileInput").click()}
      >
        Click or drag an image here
      </div>

      <input
        id="fileInput"
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => handleFile(e.target.files[0])}
      />

      {uploading && (
        <div className="progress-container">
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${progress}%` }}
            >
              <span className="progress-text">{progress}%</span>
            </div>
          </div>
        </div>
      )}

      {responseData && (
        <div className="result-container">
          {/* Image Preview Section */}
          <div className="image-preview-section">
            <h3 className="section-title">
              <span className="section-icon">📸</span>
              Vehicle Image
            </h3>
            {responseData.image && (
              <div className="image-frame">
                <img
                  src={responseData.image}
                  alt="Vehicle damage assessment"
                  className="preview-image"
                />
              </div>
            )}
          </div>

          {/* Summary Section */}
          <SummarySection responseData={responseData} />

          {/* Damages Section */}
          {responseData.damages && responseData.damages.length > 0 && (
            <div className="damages-section">
              <h3 className="section-title">
                <span className="section-icon">🔍</span>
                Damage Analysis ({responseData.damages.length} found)
              </h3>
              <div className="damages-grid">
                {responseData.damages.map((damage, index) => (
                  <DamageCard 
                    key={index} 
                    damage={damage} 
                    index={index}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
