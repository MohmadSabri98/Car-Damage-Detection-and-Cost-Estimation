import React from 'react';
import './SummarySection.css';

const SummarySection = ({ responseData }) => {
  const formatText = (text) => {
    if (text === null || text === undefined) return "";
    return String(text).replace(/[^a-zA-Z0-9\u0600-\u06FF]+/g, " ").trim();
  };

  const totalCost = parseFloat(responseData?.total_cost) || 0;
  const damages = responseData?.damages || [];
  
  // Calculate statistics
  const totalDamages = damages.length;
  const highSeverityCount = damages.filter(d => 
    d.severity?.toLowerCase().includes('high') || 
    d.severity?.toLowerCase().includes('severe')
  ).length;
  
  const normalizeConfidence = (confidence) => {
    if (!confidence) return 0;
    
    // Parse the confidence value
    let conf = parseFloat(confidence);
    
    // If confidence is between 0 and 1, convert to percentage
    if (conf <= 1 && conf > 0) {
      conf = conf * 100;
    }
    
    // Ensure it's between 0 and 100
    return Math.max(0, Math.min(100, conf));
  };

  const averageConfidence = damages.length > 0 
    ? damages.reduce((sum, d) => sum + normalizeConfidence(d.confidence), 0) / damages.length 
    : 0;

  const severityDistribution = {
    high: damages.filter(d => 
      d.severity?.toLowerCase().includes('high') || 
      d.severity?.toLowerCase().includes('severe')
    ).length,
    medium: damages.filter(d => 
      d.severity?.toLowerCase().includes('medium') || 
      d.severity?.toLowerCase().includes('moderate')
    ).length,
    low: damages.filter(d => 
      d.severity?.toLowerCase().includes('low') || 
      d.severity?.toLowerCase().includes('minor')
    ).length
  };

  return (
    <div className="summary-section">
      <div className="summary-header">
        <h2>Assessment Summary</h2>
        <div className="summary-icon">📊</div>
      </div>
      
      <div className="summary-grid">
        <div className="summary-card primary">
          <div className="summary-card-header">
            <div className="summary-icon-large">💰</div>
            <h3>Total Repair Cost</h3>
          </div>
          <div className="summary-value">
            <span className="currency-large">$</span>
            <span className="amount-large">{formatText(responseData?.total_cost)}</span>
          </div>
          <div className="summary-subtitle">Repair cost</div>
        </div>

        <div className="summary-card">
          <div className="summary-card-header">
            <div className="summary-icon-medium">🔧</div>
            <h3>Parts</h3>
          </div>
          <div className="summary-value">
            <span className="number-large">{totalDamages}</span>
          </div>
          <div className="summary-subtitle">Parts to repair</div>
        </div>

        <div className="summary-card">
          <div className="summary-card-header">
            <div className="summary-icon-medium">⚠️</div>
            <h3>Critical</h3>
          </div>
          <div className="summary-value">
            <span className="number-large">{highSeverityCount}</span>
          </div>
          <div className="summary-subtitle">High severity</div>
        </div>

        <div className="summary-card">
          <div className="summary-card-header">
            <div className="summary-icon-medium">🎯</div>
            <h3>Score</h3>
          </div>
          <div className="summary-value">
            <span className="number-large">{averageConfidence.toFixed(1)}%</span>
          </div>
          <div className="summary-subtitle">AI confidence</div>
        </div>
      </div>

      <div className="severity-breakdown">
        <h4>Severity Distribution</h4>
        <div className="severity-chart">
          <div className="severity-bar high">
            <div className="severity-label">
              <span className="severity-dot high"></span>
              High ({severityDistribution.high})
            </div>
            <div 
              className="severity-fill high"
              style={{ width: `${totalDamages > 0 ? (severityDistribution.high / totalDamages) * 100 : 0}%` }}
            ></div>
          </div>
          
          <div className="severity-bar medium">
            <div className="severity-label">
              <span className="severity-dot medium"></span>
              Medium ({severityDistribution.medium})
            </div>
            <div 
              className="severity-fill medium"
              style={{ width: `${totalDamages > 0 ? (severityDistribution.medium / totalDamages) * 100 : 0}%` }}
            ></div>
          </div>
          
          <div className="severity-bar low">
            <div className="severity-label">
              <span className="severity-dot low"></span>
              Low ({severityDistribution.low})
            </div>
            <div 
              className="severity-fill low"
              style={{ width: `${totalDamages > 0 ? (severityDistribution.low / totalDamages) * 100 : 0}%` }}
            ></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SummarySection;
