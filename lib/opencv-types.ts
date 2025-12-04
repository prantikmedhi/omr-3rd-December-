import { useEffect, useState } from "react"
import type { OpenCV } from "./opencv-types"

declare global {
  interface Window {
    cv: OpenCV
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
      if (window.cv && window.cv.getBuildInformation) {
        setLoaded(true)
      } else if (window.cv) {
        // Wait for onRuntimeInitialized
        window.cv.onRuntimeInitialized = () => {
          setLoaded(true)
        }
      }
    }
    script.onerror = () => {
      console.error("Failed to load OpenCV.js")
    }
    document.body.appendChild(script)

    return () => {
      // Cleanup not strictly necessary for single page app singleton script
    }
  }, [])

  return loaded
}
