import React, { useState } from 'react'
import './Home.css'

function Home() {
  const [message, setMessage] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const checkBackend = async () => {
    setLoading(true)
    setError(null)
    setMessage('')

    try {
      const response = await fetch('http://localhost:8000/')

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setMessage(data.message || JSON.stringify(data))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="home">
      <h1>CodePilot AI</h1>

      <button
        onClick={checkBackend}
        disabled={loading}
        className="check-button"
      >
        {loading ? 'Checking...' : 'Check Backend'}
      </button>

      {message && (
        <div className="message success">
          {message}
        </div>
      )}

      {error && (
        <div className="message error">
          Error: {error}
        </div>
      )}
    </div>
  )
}

export default Home
