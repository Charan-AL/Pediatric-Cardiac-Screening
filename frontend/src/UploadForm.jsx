import React, {useState} from 'react'

export default function UploadForm(){
  const [audio, setAudio] = useState(null)
  const [us, setUs] = useState(null)
  const [xray, setXray] = useState(null)
  const [loading, setLoading] = useState(false)
  const [report, setReport] = useState(null)
  const [error, setError] = useState(null)
  const [patientId, setPatientId] = useState('PT-001')
  const [patientAge, setPatientAge] = useState('6')

  async function onSubmit(e){
    e.preventDefault()
    setLoading(true); setError(null); setReport(null)
    try{
      const fd = new FormData()
      if(audio) fd.append('audio_file', audio)
      if(us) fd.append('us_file', us)
      if(xray) fd.append('xray_file', xray)

      const res = await fetch('http://localhost:8000/predict', { method: 'POST', body: fd })
      if(!res.ok){
        const txt = await res.text()
        throw new Error(`API error: ${res.status} ${txt}`)
      }
      const json = await res.json()
      if(json.has_gradcam && json.report_url){
        const r = await fetch(`http://localhost:8000${json.report_url}`)
        const full = await r.json()
        setReport({...json, gradcam_images: full.gradcam_images})
      } else {
        setReport(json)
      }
    }catch(err){
      setError(err.message)
    }finally{
      setLoading(false)
    }
  }

  return (
    <div className="upload-wrapper">
      <form className="upload-panel" onSubmit={onSubmit}>
        <h2>📋 Upload Modalities</h2>
        
        <div className="patient-info">
          <div className="form-group">
            <label>Patient ID</label>
            <input type="text" value={patientId} onChange={e=>setPatientId(e.target.value)} />
          </div>
          <div className="form-group">
            <label>Age (months)</label>
            <input type="number" value={patientAge} onChange={e=>setPatientAge(e.target.value)} />
          </div>
        </div>

        <div className="form-group">
          <label>🔊 Heart Sound (WAV)</label>
          <input type="file" accept="audio/wav" onChange={e=>setAudio(e.target.files[0])} />
          {audio && <span className="file-name">✓ {audio.name}</span>}
        </div>

        <div className="form-group">
          <label>🖥️ Ultrasound (JPG/PNG)</label>
          <input type="file" accept="image/*" onChange={e=>setUs(e.target.files[0])} />
          {us && <span className="file-name">✓ {us.name}</span>}
        </div>

        <div className="form-group">
          <label>🩻 Chest X-Ray (JPG/PNG)</label>
          <input type="file" accept="image/*" onChange={e=>setXray(e.target.files[0])} />
          {xray && <span className="file-name">✓ {xray.name}</span>}
        </div>

        <button type="submit" disabled={loading} className="btn-submit">
          {loading? '⏳ Running Screening...' : '▶ Run Screening'}
        </button>

        {error && <div className="error-box">{error}</div>}
      </form>

      {report && (
        <div className="results-panel">
          <div className={`decision-badge ${report.decision.toLowerCase()}`}>
            {report.decision === 'REFER' 
              ? `🚨 REFER — ${(report.probability_of_chd*100).toFixed(1)}% CHD Probability`
              : `✅ PASS — ${((1-report.probability_of_chd)*100).toFixed(1)}% Normal`
            }
          </div>

          <div className="info-section">
            <label>Patient: {patientId} ({patientAge}mo)</label>
            <div className="progress-bar">
              <div className="progress-fill" style={{width: `${report.probability_of_chd*100}%`}}></div>
              <span className="progress-text">CHD Probability: {(report.probability_of_chd*100).toFixed(1)}%</span>
            </div>
          </div>

          <div className="reliability-section">
            <h3>🎛️ Modality Reliability (Gate Weights)</h3>
            {report.modality_reliability && (
              <div className="reliability-grid">
                {Object.entries(report.modality_reliability).map(([mod, val]) => (
                  <div key={mod} className="reliability-item">
                    <label>{mod.replace('_', ' ').toUpperCase()}</label>
                    <div className="bar-container">
                      <div className="bar" style={{width: `${val*100}%`}}></div>
                    </div>
                    <span className="value">{val.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {report.gradcam_images && Object.keys(report.gradcam_images).length > 0 && (
            <div className="gradcam-section">
              <h3>🔬 Grad-CAM Explainability</h3>
              <div className="gradcam-grid">
                {report.gradcam_images.audio_gradcam && (
                  <div className="gradcam-item">
                    <h4>🔊 Heart Sound Saliency</h4>
                    <img src={`data:image/png;base64,${report.gradcam_images.audio_gradcam}`} alt="audio" />
                  </div>
                )}
                {report.gradcam_images.ultrasound_gradcam && (
                  <div className="gradcam-item">
                    <h4>🖥️ Ultrasound Attention</h4>
                    <img src={`data:image/png;base64,${report.gradcam_images.ultrasound_gradcam}`} alt="us" />
                  </div>
                )}
                {report.gradcam_images.xray_gradcam && (
                  <div className="gradcam-item">
                    <h4>🩻 X-Ray Attention</h4>
                    <img src={`data:image/png;base64,${report.gradcam_images.xray_gradcam}`} alt="xray" />
                  </div>
                )}
              </div>
            </div>
          )}

          <div className="advice-section">
            <h3>📝 Clinical Advice</h3>
            <p className={`advice ${report.decision.toLowerCase()}`}>{report.advice}</p>
            <p className="disclaimer">⚠️ This is an AI-assisted screening tool. It does NOT replace clinical judgement or formal diagnostic evaluation.</p>
          </div>
        </div>
      )}
    </div>
  )
}
