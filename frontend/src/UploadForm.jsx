import React, {useState} from 'react'

export default function UploadForm(){
  const [audio, setAudio] = useState(null)
  const [us, setUs] = useState(null)
  const [xray, setXray] = useState(null)
  const [loading, setLoading] = useState(false)
  const [report, setReport] = useState(null)
  const [error, setError] = useState(null)

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
      // if there are gradcam images, fetch the full report
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
    <form className="card" onSubmit={onSubmit}>
      <label>Heart sound (WAV)</label>
      <input type="file" accept="audio/wav" onChange={e=>setAudio(e.target.files[0])} />

      <label>Ultrasound (JPG/PNG)</label>
      <input type="file" accept="image/*" onChange={e=>setUs(e.target.files[0])} />

      <label>Chest X-ray (JPG/PNG)</label>
      <input type="file" accept="image/*" onChange={e=>setXray(e.target.files[0])} />

      <button type="submit" disabled={loading}>
        {loading? 'Running...' : 'Run Screening'}
      </button>

      {error && <div className="error">{error}</div>}

      {report && (
        <div className="result">
          <h2>{report.decision} — {report.probability_of_chd*100}%</h2>
          <p><strong>Confidence:</strong> {report.confidence}</p>
          <h3>Modality Reliability</h3>
          <pre>{JSON.stringify(report.modality_reliability, null, 2)}</pre>

          {report.gradcam_images && (
            <div className="images">
              {report.gradcam_images.audio_gradcam && (
                <div>
                  <h4>Audio Grad-CAM</h4>
                  <img src={`data:image/png;base64,${report.gradcam_images.audio_gradcam}`} alt="audio" />
                </div>
              )}
              {report.gradcam_images.ultrasound_gradcam && (
                <div>
                  <h4>Ultrasound Grad-CAM</h4>
                  <img src={`data:image/png;base64,${report.gradcam_images.ultrasound_gradcam}`} alt="us" />
                </div>
              )}
              {report.gradcam_images.xray_gradcam && (
                <div>
                  <h4>X-ray Grad-CAM</h4>
                  <img src={`data:image/png;base64,${report.gradcam_images.xray_gradcam}`} alt="xray" />
                </div>
              )}
            </div>
          )}

          <p className="advice">{report.advice}</p>
        </div>
      )}
    </form>
  )
}
