import React from 'react';
import './DamageCard.css';

const DamageCard = ({ damage, index }) => {
  const formatText = (text) => {
    if (text === null || text === undefined) return "";
    return String(text).replace(/[^a-zA-Z0-9\u0600-\u06FF]+/g, " ").trim();
  };

  const getSeverityColor = (severity) => {
    const severityLower = severity?.toLowerCase() || '';
    if (severityLower.includes('high') || severityLower.includes('severe')) return '#dc3545';
    if (severityLower.includes('medium') || severityLower.includes('moderate')) return '#ffc107';
    if (severityLower.includes('low') || severityLower.includes('minor')) return '#28a745';
    return '#6c757d';
  };

  const getSeverityIcon = (severity) => {
    const severityLower = severity?.toLowerCase() || '';
    if (severityLower.includes('high') || severityLower.includes('severe')) return '🔴';
    if (severityLower.includes('medium') || severityLower.includes('moderate')) return '🟡';
    if (severityLower.includes('low') || severityLower.includes('minor')) return '🟢';
    return '⚪';
  };

  const getSeverityClass = (severity) => {
    const severityLower = severity?.toLowerCase() || '';
    if (severityLower.includes('high') || severityLower.includes('severe')) return 'high';
    if (severityLower.includes('medium') || severityLower.includes('moderate')) return 'medium';
    if (severityLower.includes('low') || severityLower.includes('minor')) return 'low';
    return 'unknown';
  };

  const getSeverityLabel = (severity) => {
    const severityLower = severity?.toLowerCase() || '';
    if (severityLower.includes('high') || severityLower.includes('severe')) return 'High';
    if (severityLower.includes('medium') || severityLower.includes('moderate')) return 'Medium';
    if (severityLower.includes('low') || severityLower.includes('minor')) return 'Low';
    return 'Unknown';
  };

  const getConfidenceColor = (confidence) => {
    const conf = parseFloat(confidence) || 0;
    if (conf >= 80) return '#28a745';
    if (conf >= 60) return '#ffc107';
    return '#dc3545';
  };

  const normalizeConfidence = (confidence) => {
    if (!confidence) return 0;
    
    // Debug: Log the original confidence value
    console.log('Original confidence value:', confidence, 'Type:', typeof confidence);
    
    // Parse the confidence value
    let conf = parseFloat(confidence);
    
    // If confidence is between 0 and 1, convert to percentage
    if (conf <= 1 && conf > 0) {
      conf = conf * 100;
    }
    
    // Ensure it's between 0 and 100
    const normalized = Math.max(0, Math.min(100, conf));
    console.log('Normalized confidence:', normalized);
    
    return normalized;
  };

  return (
    <div className="damage-card">
      <div className="damage-card-header">
        <div className="damage-number">#{index + 1}</div>
        <div className="damage-part">
          <span className="part-icon">🔧</span>
          {formatText(damage.part)}
        </div>
      </div>
      
      <div className="damage-card-content">
        <div className="damage-metrics">
          <div className="metric-item severity">
            <div className="metric-label">Severity</div>
            <div className="metric-value severity-value">
              <div className="severity-display">
                <span className={`severity-dot ${getSeverityClass(damage.severity)}`}></span>
                {getSeverityLabel(damage.severity)}
              </div>
            </div>
          </div>

          <div className="metric-item confidence">
            <div className="metric-label">Confidence</div>
            <div 
              className="metric-value confidence-value"
              style={{ color: getConfidenceColor(normalizeConfidence(damage.confidence)) }}
            >
              <div className="confidence-bar">
                <div 
                  className="confidence-fill"
                  style={{ 
                    width: `${normalizeConfidence(damage.confidence)}%`,
                    backgroundColor: getConfidenceColor(normalizeConfidence(damage.confidence))
                  }}
                ></div>
              </div>
              <span className="confidence-text">{normalizeConfidence(damage.confidence).toFixed(1)}%</span>
            </div>
          </div>

          <div className="metric-item cost">
            <div className="metric-label">Repair Cost</div>
            <div className="metric-value cost-value">
              <span className="currency">$</span>
              <span className="cost-amount">{formatText(damage.cost)}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="damage-card-footer">
        <div className="damage-indicator" style={{ backgroundColor: damage.color || '#007bff' }}></div>
      </div>
    </div>
  );
};

export default DamageCard;
