import { useEffect, useState } from "react"

declare global {
  interface Window {
    cv: any
  }
}

export function useOpenCV() {
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    if (window.cv) {
      setLoaded(true)
      return
    }

    const script = document.createElement("script")
    script.src = "/js/opencv.js"
    script.async = true
    script.onload = () => {
      // OpenCV.js sometimes takes a moment to initialize even after load
      if (window.cv.getBuildInformation) {
        setLoaded(true)
      } else {
        // Wait for onRuntimeInitialized
        window.cv.onRuntimeInitialized = () => {
          setLoaded(true)
        }
      }
    }
    document.body.appendChild(script)

    return () => {
      // Cleanup not strictly necessary for single page app singleton script
    }
  }, [])

  return loaded
}
