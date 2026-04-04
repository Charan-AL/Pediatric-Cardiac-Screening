import React from 'react'
import UploadForm from './UploadForm'

export default function App() {
  return (
    <div className="container">
      <h1>🫀 Pediatric Cardiac Screening</h1>
      <p className="muted">Upload a WAV and/or ultrasound / x-ray images.</p>
      <UploadForm />
    </div>
  )
}
