import React from 'react';
import './Header.css';

const Header = () => {
  return (
    <header className="header">
      <div className="header-content">
        <div className="header-title">
          <h1>Car Damage Detection & Cost Estimator</h1>
          <p className="header-subtitle">AI-powered vehicle damage assessment and repair cost estimation</p>
        </div>
        <div className="header-icon">
          <div className="car-icon">🚗</div>
        </div>
      </div>
    </header>
  );
};

export default Header;

