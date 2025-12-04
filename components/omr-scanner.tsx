"use client"

import type React from "react"

import { useState, useCallback, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Switch } from "@/components/ui/switch"
import { Upload, FileSpreadsheet, Loader2, CheckCircle, XCircle, AlertCircle, MinusCircle, Camera, RotateCcw } from "lucide-react"
import { processOMRSheet, type OMRResult } from "@/lib/omr-processor"
import { parseAnswerKey } from "@/lib/answer-key-parser"
import { useOpenCV } from "@/lib/use-opencv"
import { ModeToggle } from "@/components/mode-toggle"

export default function OMRScanner() {

const profiles = [
  {
    name: "Prantik Medhi",
    url: "https://www.linkedin.com/in/prantikmedhi",
  },
  {
    name: "Duke Bhuyan Borah",
    url: "https://in.linkedin.com/in/duke3huyan3orah",
  },
];

// Randomize on each refresh
const randomProfiles = [...profiles].sort(() => Math.random() - 0.5);
  const opencvLoaded = useOpenCV()
  const [omrSheet, setOmrSheet] = useState<File | null>(null)
  const [answerKey, setAnswerKey] = useState<File | null>(null)
  const [choicesPerQuestion, setChoicesPerQuestion] = useState(4)
  const [minFillThreshold, setMinFillThreshold] = useState(0.25)
  const [binaryThreshold, setBinaryThreshold] = useState(128)
  const [showThresholdedImage, setShowThresholdedImage] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [result, setResult] = useState<OMRResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [omrPreview, setOmrPreview] = useState<string | null>(null)
  const [thresholdedPreview, setThresholdedPreview] = useState<string | null>(null)
  const [showCamera, setShowCamera] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const handleOmrChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null
    setOmrSheet(file)
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => setOmrPreview(e.target?.result as string)
      reader.readAsDataURL(file)
    } else {
      setOmrPreview(null)
    }
    // Clear previous results/previews
    setResult(null)
    setThresholdedPreview(null)
  }

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" }
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
      setShowCamera(true)
    } catch (err) {
      console.error("Error accessing camera:", err)
      setError("Could not access camera. Please check permissions.")
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    setShowCamera(false)
  }

  const captureImage = () => {
    if (videoRef.current) {
      const canvas = document.createElement("canvas")
      canvas.width = videoRef.current.videoWidth
      canvas.height = videoRef.current.videoHeight
      const ctx = canvas.getContext("2d")
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0)
        canvas.toBlob((blob) => {
          if (blob) {
            const file = new File([blob], "captured-omr.png", { type: "image/png" })
            setOmrSheet(file)
            setOmrPreview(canvas.toDataURL("image/png"))
            setResult(null)
            setThresholdedPreview(null)
            stopCamera()
          }
        }, "image/png")
      }
    }
  }

  // Effect to update threshold preview when slider changes
  useEffect(() => {
    let timeoutId: NodeJS.Timeout

    if (opencvLoaded && omrSheet && showThresholdedImage) {
      // Debounce to avoid blocking UI
      timeoutId = setTimeout(async () => {
        try {
          // Re-processing the whole thing might be heavy.
          // Let's stick to "Process" button for now, but maybe auto-process if result exists?
          if (result && answerKey) {
             handleProcess()
          }
        } catch (e) {
          console.error(e)
        }
      }, 500)
    }
    return () => clearTimeout(timeoutId)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [binaryThreshold, choicesPerQuestion, minFillThreshold])

  const handleProcess = useCallback(async () => {
    if (!opencvLoaded) {
      setError("OpenCV is still loading...")
      return
    }
    if (!omrSheet || !answerKey) {
      setError("Please select both the OMR sheet and the answer key.")
      return
    }

    setProcessing(true)
    setError(null)
    // Don't clear result immediately to avoid flicker if just updating threshold

    try {
      // Parse answer key
      const answerKeyText = await answerKey.text()
      const answers = parseAnswerKey(answerKeyText)

      if (Object.keys(answers).length === 0) {
        throw new Error("Could not parse any answers from the answer key file.")
      }

      // Process OMR sheet
      const processResult = await processOMRSheet(
        omrSheet,
        answers,
        choicesPerQuestion,
        minFillThreshold,
        binaryThreshold
      )

      setResult(processResult)
      setThresholdedPreview(processResult.processedImage) // The processor returns the annotated image, maybe we want the binary one?
      // We will update the processor to return what we need.

    } catch (err) {
      console.error("Processing error:", err)
      setError(err instanceof Error ? err.message : "An error occurred while processing the OMR sheet.")
    } finally {
      setProcessing(false)
    }
  }, [omrSheet, answerKey, choicesPerQuestion, minFillThreshold, binaryThreshold, opencvLoaded])

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "correct":
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case "incorrect":
        return <XCircle className="h-4 w-4 text-red-500" />
      case "unanswered":
        return <MinusCircle className="h-4 w-4 text-orange-500" />
      case "multiple":
        return <AlertCircle className="h-4 w-4 text-yellow-500" />
      default:
        return null
    }
  }

  return (
    <div className="container mx-auto p-6 max-w-6xl flex flex-col min-h-screen">
      <div className="flex justify-end mb-4">
        <ModeToggle />
      </div>
      <div className="mb-8 text-center">
        <h1 className="text-4xl font-bold text-foreground mb-2">OMR Sheet Scanner</h1>
        <p className="text-muted-foreground">
            {!opencvLoaded ? (
                <span className="flex items-center justify-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Loading OpenCV...
                </span>
            ) : (
                "Upload your OMR sheet and answer key to get instant results"
            )}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Section */}
        <Card>
          <CardHeader>
            <CardTitle>Upload Files</CardTitle>
            <CardDescription>Select your OMR sheet image and answer key CSV file</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* OMR Sheet Upload */}
            <div className="space-y-2">
              <Label htmlFor="omr-sheet" className="flex items-center gap-2">
                <Upload className="h-4 w-4" />
                OMR Sheet Image
              </Label>
              <div className="flex gap-2">
                 <Input
                    id="omr-sheet"
                    type="file"
                    accept="image/*"
                    onChange={handleOmrChange}
                    className="cursor-pointer flex-1"
                  />
                  <Button variant="outline" size="icon" onClick={startCamera} title="Capture from Camera">
                    <Camera className="h-4 w-4" />
                  </Button>
              </div>

              {showCamera && (
                <div className="mt-2 border rounded-lg overflow-hidden relative bg-black">
                   <video ref={videoRef} autoPlay playsInline className="w-full h-64 object-contain" />
                   <div className="absolute bottom-4 left-0 right-0 flex justify-center gap-4">
                     <Button onClick={captureImage} variant="default">Capture</Button>
                     <Button onClick={stopCamera} variant="secondary">Cancel</Button>
                   </div>
                </div>
              )}

              {omrPreview && !showCamera && (
                <div className="mt-2 border rounded-lg overflow-hidden relative">
                  <img
                    src={showThresholdedImage && result?.processedImage ? result.processedImage : omrPreview}
                    alt="OMR Preview"
                    className="w-full h-48 object-contain bg-muted"
                  />
                  {result && (
                      <div className="absolute top-2 right-2 bg-black/50 text-white text-xs px-2 py-1 rounded">
                          {showThresholdedImage ? "Processed View" : "Original View"}
                      </div>
                  )}
                </div>
              )}
            </div>

            {/* Answer Key Upload */}
            <div className="space-y-2">
              <Label htmlFor="answer-key" className="flex items-center gap-2">
                <FileSpreadsheet className="h-4 w-4" />
                Answer Key (CSV)
              </Label>
              <Input
                id="answer-key"
                type="file"
                accept=".csv"
                onChange={(e) => setAnswerKey(e.target.files?.[0] || null)}
                className="cursor-pointer"
              />
              <p className="text-xs text-muted-foreground">
                CSV format: question_number, correct_answer_index (1-based)
              </p>
            </div>

            {/* Settings */}
            <div className="space-y-4 pt-4 border-t">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                   <Label>Choices per Question</Label>
                   <Input
                      type="number"
                      value={choicesPerQuestion}
                      onChange={(e) => setChoicesPerQuestion(Number(e.target.value))}
                      className="w-20 h-8"
                      min={2}
                      max={10}
                   />
                </div>
                <Slider
                  value={[choicesPerQuestion]}
                  onValueChange={(v) => setChoicesPerQuestion(v[0])}
                  min={2}
                  max={10}
                  step={1}
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                    <Label>Binarization Threshold</Label>
                    <div className="flex items-center space-x-2">
                        <Input
                           type="number"
                           value={binaryThreshold}
                           onChange={(e) => setBinaryThreshold(Number(e.target.value))}
                           className="w-20 h-8"
                           min={0}
                           max={255}
                        />
                        <Switch
                            id="show-threshold"
                            checked={showThresholdedImage}
                            onCheckedChange={setShowThresholdedImage}
                            disabled={!result}
                        />
                        <Label htmlFor="show-threshold" className="text-xs font-normal">Show Processed</Label>
                    </div>
                </div>
                <Slider
                  value={[binaryThreshold]}
                  onValueChange={(v) => setBinaryThreshold(v[0])}
                  min={0}
                  max={255}
                  step={1}
                />
                <p className="text-xs text-muted-foreground">Adjust if bubbles are not detected correctly. Lower is darker.</p>
              </div>

              <div className="space-y-2">
                 <div className="flex items-center justify-between">
                   <Label>Bubble Fill Threshold</Label>
                   <Input
                      type="number"
                      value={minFillThreshold}
                      onChange={(e) => setMinFillThreshold(Number(e.target.value))}
                      className="w-20 h-8"
                      min={0.1}
                      max={0.9}
                      step={0.05}
                   />
                </div>
                <Label className="text-xs text-muted-foreground">{(minFillThreshold * 100).toFixed(0)}%</Label>
                <Slider
                  value={[minFillThreshold]}
                  onValueChange={(v) => setMinFillThreshold(v[0])}
                  min={0.1}
                  max={0.9}
                  step={0.05}
                />
              </div>
            </div>

            {/* Process Button */}
            <Button
              onClick={handleProcess}
              disabled={processing || !omrSheet || !answerKey || !opencvLoaded}
              className="w-full"
              size="lg"
            >
              {processing ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Processing...
                </>
              ) : (
                "Process OMR Sheet"
              )}
            </Button>

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {/* Results Section */}
        <Card>
          <CardHeader>
            <CardTitle>Results</CardTitle>
            <CardDescription>
              {result ? "Processing complete" : "Upload files and click process to see results"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {result ? (
              <div className="space-y-6">
                {/* Score Summary */}
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-green-50 dark:bg-green-950 p-4 rounded-lg text-center">
                    <div className="text-3xl font-bold text-green-600 dark:text-green-400">
                      {result.score}/{result.total}
                    </div>
                    <div className="text-sm text-green-700 dark:text-green-300">Score</div>
                  </div>
                  <div className="bg-purple-50 dark:bg-purple-950 p-4 rounded-lg text-center">
                    <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">
                      {result.totalMarks}
                    </div>
                    <div className="text-sm text-purple-700 dark:text-purple-300">Total Marks</div>
                  </div>
                  <div className="bg-blue-50 dark:bg-blue-950 p-4 rounded-lg text-center">
                    <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">
                      {((result.score / result.total) * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-blue-700 dark:text-blue-300">Percentage</div>
                  </div>
                </div>

                {/* Stats */}
                <div className="flex gap-4 text-sm">
                  <div className="flex items-center gap-1">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    <span>Correct: {result.score}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <XCircle className="h-4 w-4 text-red-500" />
                    <span>Wrong: {result.total - result.score - result.unanswered - result.multipleAnswers}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <MinusCircle className="h-4 w-4 text-orange-500" />
                    <span>Blank: {result.unanswered}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <AlertCircle className="h-4 w-4 text-yellow-500" />
                    <span>Multiple: {result.multipleAnswers}</span>
                  </div>
                </div>

                {/* Processed Image */}
                {result.processedImage && (
                  <div className="border rounded-lg overflow-hidden">
                    <img
                      src={result.processedImage || "/placeholder.svg"}
                      alt="Processed OMR Sheet"
                      className="w-full object-contain"
                    />
                  </div>
                )}

                {/* Question Details */}
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  <h3 className="font-semibold text-sm">Question Details</h3>
                  <div className="grid gap-2">
                    {result.results.map((r) => (
                      <div key={r.question} className="flex items-center justify-between p-2 bg-muted rounded text-sm">
                        <div className="flex items-center gap-2">
                          {getStatusIcon(r.status)}
                          <span>Q{r.question}</span>
                        </div>
                        <div className="flex gap-4 text-muted-foreground">
                          <span>Selected: {r.selected !== null ? String.fromCharCode(65 + r.selected) : "-"}</span>
                          <span>
                            Correct: {r.correctAnswer !== null ? String.fromCharCode(65 + r.correctAnswer) : "-"}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-64 text-muted-foreground">
                <FileSpreadsheet className="h-12 w-12 mb-4" />
                <p>No results yet</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
      <footer className="mt-auto py-6 text-center text-sm text-muted-foreground border-t">
  <p>
    Made by{" "}
    <a
      href={randomProfiles[0].url}
      target="_blank"
      rel="noopener noreferrer"
      className="font-medium underline underline-offset-4 hover:text-primary"
    >
      {randomProfiles[0].name}
    </a>{" "}
    &{" "}
    <a
      href={randomProfiles[1].url}
      target="_blank"
      rel="noopener noreferrer"
      className="font-medium underline underline-offset-4 hover:text-primary"
    >
      {randomProfiles[1].name}
    </a>
  </p>
</footer>
    </div>
  )
}
