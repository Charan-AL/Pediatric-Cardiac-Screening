import React from 'react'
import UploadForm from './UploadForm'
import ModelMetrics from './ModelMetrics'

export default function App() {
  const [showMetrics, setShowMetrics] = React.useState(false)

  return (
    <div className="container">
      <h1>🫀 Autonomous Pediatric Cardiac Screening</h1>
      <p className="muted">Upload a WAV and/or ultrasound / x-ray images. AI will analyze all available modalities.</p>
      {/* PHASE 1 (50%): Currently showing only audio + ultrasound models. Phase 2 will add X-ray + fusion. */}
      
      <div style={{ marginBottom: '20px' }}>
        <button 
          onClick={() => setShowMetrics(!showMetrics)}
          style={{ 
            padding: '8px 16px', 
            background: '#457b9d', 
            color: 'white', 
            border: 'none', 
            borderRadius: '6px',
            cursor: 'pointer'
          }}
        >
          {showMetrics ? '👉 Hide' : '📊 Show'} Model Performance Proof
        </button>
      </div>

      {showMetrics && <ModelMetrics />}
      
      <hr style={{ borderColor: '#457b9d', margin: '20px 0' }} />
      
      <UploadForm />
    </div>
  )
}
