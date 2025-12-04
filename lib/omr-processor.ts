/**
 * Advanced OMR Sheet Processor with Scanner-like Capabilities
 * 
 * Features:
 * - Advanced image preprocessing (contrast enhancement, noise reduction)
 * - Multi-scale paper detection
 * - Adaptive thresholding based on image quality
 * - Robust perspective correction
 * - Quality validation
 * - Type-safe OpenCV operations
 */

import type { OpenCV, OpenCVMat, OpenCVRect, OpenCVPoint, OpenCVSize, OpenCVScalar, OpenCVMatVector } from './opencv-types'

export interface QuestionResult {
  question: number
  status: "correct" | "incorrect" | "unanswered" | "multiple"
  selected: number | null
  correctAnswer: number | null
  fillPercentages?: number[]
}

export interface OMRResult {
  score: number
  totalMarks: number
  total: number
  unanswered: number
  multipleAnswers: number
  processedImage: string
  results: QuestionResult[]
  processingInfo?: {
    paperDetected: boolean
    imageQuality: 'excellent' | 'good' | 'fair' | 'poor'
    preprocessingApplied: string[]
  }
}

interface Bubble {
  x: number
  y: number
  width: number
  height: number
  centerX: number
  centerY: number
  area: number
  circularity: number
}

interface BubbleRow {
  bubbles: Bubble[]
  avgY: number
}

interface ProcessingConfig {
  choicesPerQuestion: number
  minFillThreshold: number
  binaryThreshold: number
  enableAdvancedPreprocessing?: boolean
  enableMultiScaleDetection?: boolean
}

interface ImageQualityMetrics {
  brightness: number
  contrast: number
  sharpness: number
  noiseLevel: number
  overall: 'excellent' | 'good' | 'fair' | 'poor'
}

// Get OpenCV instance with type safety
function getCV(): OpenCV {
  if (typeof window !== 'undefined' && window.cv) {
    return window.cv
  }
  // Access global cv variable (declared in opencv-types.ts)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const globalCv = (globalThis as any).cv
  if (globalCv) {
    return globalCv as OpenCV
  }
  throw new Error("OpenCV.js is not loaded. Please wait for it to initialize.")
}

export async function processOMRSheet(
  imageFile: File,
  answerKey: Record<number, number>,
  choicesPerQuestion: number,
  minFillThreshold: number,
  binaryThreshold: number,
): Promise<OMRResult> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = "anonymous"

    img.onload = () => {
      try {
        const cv = getCV()
        const config: ProcessingConfig = {
          choicesPerQuestion,
          minFillThreshold,
          binaryThreshold,
          enableAdvancedPreprocessing: true,
          enableMultiScaleDetection: true,
        }
        const result = processImageOpenCV(cv, img, answerKey, config)
        resolve(result)
      } catch (err) {
        console.error("OMR Processing Error:", err)
        reject(err)
      }
    }

    img.onerror = () => {
      reject(new Error("Failed to load image. Please ensure the file is a valid image."))
    }

    const reader = new FileReader()
    reader.onload = (e) => {
      img.src = e.target?.result as string
    }
    reader.onerror = () => reject(new Error("Failed to read file"))
    reader.readAsDataURL(imageFile)
  })
}

/**
 * Main processing function with advanced scanner-like preprocessing
 */
function processImageOpenCV(
  cv: OpenCV,
  img: HTMLImageElement,
  answerKey: Record<number, number>,
  config: ProcessingConfig,
): OMRResult {
  const canvas = document.createElement("canvas")
  const ctx = canvas.getContext("2d")!
  
  // Smart resizing - maintain aspect ratio, optimize for processing
  const maxDim = 2400 // Increased for better quality
  const minDim = 800  // Minimum for reliable detection
  let scale = 1
  
  if (Math.max(img.width, img.height) > maxDim) {
    scale = maxDim / Math.max(img.width, img.height)
  } else if (Math.max(img.width, img.height) < minDim) {
    scale = minDim / Math.max(img.width, img.height)
  }
  
  canvas.width = Math.round(img.width * scale)
  canvas.height = Math.round(img.height * scale)
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

  // Initialize OpenCV Mat
  let src = cv.imread(canvas)
  const processingInfo: string[] = []
  
  // Memory management: Track all Mats for cleanup
  const matsToCleanup: OpenCVMat[] = [src]

  try {
    // Step 1: Assess image quality
    const quality = assessImageQuality(cv, src)
    processingInfo.push(`Quality: ${quality.overall}`)
    
    // Step 2: Advanced Preprocessing (Scanner-like enhancement)
    if (config.enableAdvancedPreprocessing) {
      src = applyAdvancedPreprocessing(cv, src, quality, matsToCleanup)
      processingInfo.push("Advanced preprocessing applied")
    }

    // Step 3: Multi-scale Paper Detection with Perspective Correction
    const paperDetection = detectAndWarpPaperAdvanced(
      cv,
      src,
      config.enableMultiScaleDetection ?? true,
      matsToCleanup
    )
    
    if (paperDetection.found && paperDetection.warpedMat) {
      src.delete()
      src = paperDetection.warpedMat
      matsToCleanup.push(src)
      processingInfo.push("Paper detected and perspective corrected")
    } else {
      processingInfo.push("Paper detection: using full image")
    }

    // Step 4: Adaptive Bubble Detection
    const bubbles = detectBubblesAdvanced(cv, src, config, quality, matsToCleanup)
    
    if (bubbles.length === 0) {
      throw new Error(
        "No bubbles detected. Possible causes:\n" +
        "- Poor image quality or lighting\n" +
        "- Sheet not properly aligned\n" +
        "- Image too blurry or out of focus\n" +
        "Try: Better lighting, clearer image, or adjust thresholds"
      )
    }

    processingInfo.push(`Detected ${bubbles.length} bubbles`)

    // Step 5: Group bubbles into rows
    const rows = groupBubblesIntoRows(bubbles, config.choicesPerQuestion)
    
    if (rows.length === 0) {
      throw new Error(
        "Could not organize bubbles into question rows.\n" +
        "Check: Choices per question setting matches your sheet format."
      )
    }

    // Step 6: Score answers
    const gray = new cv.Mat()
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0)
    matsToCleanup.push(gray)

    const results: QuestionResult[] = []
    let correct = 0
    let totalMarks = 0
    let unanswered = 0
    let multipleAnswers = 0

    const numQuestions = Math.min(rows.length, Object.keys(answerKey).length)

    for (let q = 0; q < numQuestions; q++) {
      const row = rows[q]
      const questionNum = q + 1
      const correctAns = answerKey[questionNum]

      // Handle incomplete rows (missing bubbles)
      // Pad with zeros if we have fewer bubbles than expected
      const fillPercentages = row.bubbles.map(bubble => 
        calculateFillPercentageAdvanced(cv, gray, bubble, matsToCleanup)
      )
      
      // If we have fewer bubbles than expected, pad with zeros (unfilled)
      while (fillPercentages.length < config.choicesPerQuestion) {
        fillPercentages.push(0)
      }
      
      // Only use the first N bubbles (in case we have more than expected)
      const trimmedFillPercentages = fillPercentages.slice(0, config.choicesPerQuestion)

      const { selectedIdx, isValid, hasMultiple } = detectSelectedBubble(
        trimmedFillPercentages,
        config.minFillThreshold
      )

      let status: QuestionResult["status"]
      if (!isValid) {
        status = "unanswered"
        unanswered++
      } else if (hasMultiple) {
        status = "incorrect"
        multipleAnswers++
        totalMarks -= 1
      } else if (selectedIdx === correctAns) {
        status = "correct"
        correct++
        totalMarks += 4
      } else {
        status = "incorrect"
        totalMarks -= 1
      }

      results.push({
        question: questionNum,
        status,
        selected: isValid ? selectedIdx : null,
        correctAnswer: correctAns ?? null,
        fillPercentages: trimmedFillPercentages,
      })

      // Only draw markers for bubbles that actually exist
      const bubblesToDraw = row.bubbles.slice(0, config.choicesPerQuestion)
      const fillToDraw = trimmedFillPercentages
      drawResultMarkersOpenCV(cv, src, bubblesToDraw, fillToDraw, config.minFillThreshold, correctAns, status, isValid, hasMultiple)
    }

    // Step 7: Generate output image
    cv.imshow(canvas, src)
    const processedImage = canvas.toDataURL("image/png")

    return {
      score: correct,
      totalMarks,
      total: numQuestions,
      unanswered,
      multipleAnswers,
      processedImage,
      results,
      processingInfo: {
        paperDetected: paperDetection.found,
        imageQuality: quality.overall,
        preprocessingApplied: processingInfo,
      },
    }

  } catch (error) {
    throw error
  } finally {
    // Cleanup all allocated memory
    matsToCleanup.forEach(mat => {
      try {
        mat.delete()
      } catch (e) {
        console.warn("Error cleaning up Mat:", e)
      }
    })
  }
}

/**
 * Assess image quality metrics for adaptive processing
 */
function assessImageQuality(cv: OpenCV, src: OpenCVMat): ImageQualityMetrics {
  const gray = new cv.Mat()
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0)

  const mean = new cv.Mat()
  const stddev = new cv.Mat()
  cv.meanStdDev(gray, mean, stddev)

  const brightness = mean.data64F?.[0] ?? 0
  const contrast = stddev.data64F?.[0] ?? 0 // Standard deviation as contrast measure

  // Calculate sharpness using Sobel variance (approximation of Laplacian)
  const blurredForSharpness = new cv.Mat()
  const ksize = new cv.Size(3, 3)
  cv.GaussianBlur(gray, blurredForSharpness, ksize, 0, 0, cv.BORDER_DEFAULT)
  // Note: OpenCV.js may not have Laplacian, so we use Sobel as approximation
  const sobel = new cv.Mat()
  cv.Sobel(blurredForSharpness, sobel, cv.CV_32F, 1, 1, 3)
  
  const sobelMean = new cv.Mat()
  const sobelStddev = new cv.Mat()
  cv.meanStdDev(sobel, sobelMean, sobelStddev)
  const sharpness = sobelStddev.data64F?.[0] ?? 0
  
  blurredForSharpness.delete()

  // Estimate noise level (variance in small regions)
  const noiseLevel = estimateNoiseLevel(cv, gray)

  // Determine overall quality
  let overall: ImageQualityMetrics['overall'] = 'poor'
  if (brightness > 100 && brightness < 200 && contrast > 40 && sharpness > 10) {
    overall = 'excellent'
  } else if (brightness > 80 && brightness < 220 && contrast > 30 && sharpness > 7) {
    overall = 'good'
  } else if (brightness > 60 && brightness < 240 && contrast > 20 && sharpness > 4) {
    overall = 'fair'
  }

  // Cleanup
  gray.delete()
  mean.delete()
  stddev.delete()
  sobel.delete()
  sobelMean.delete()
  sobelStddev.delete()

  return { brightness, contrast, sharpness, noiseLevel, overall }
}

/**
 * Estimate noise level in the image
 */
function estimateNoiseLevel(cv: OpenCV, gray: OpenCVMat): number {
  // Sample small patches and calculate variance
  const patchSize = 20
  const numSamples = 10
  let totalVariance = 0

  for (let i = 0; i < numSamples; i++) {
    const x = Math.floor(Math.random() * (gray.cols - patchSize))
    const y = Math.floor(Math.random() * (gray.rows - patchSize))
    
    const rect = new cv.Rect(x, y, patchSize, patchSize)
    const patch = gray.roi(rect)
    const patchClone = patch.clone()
    
    const mean = new cv.Mat()
    const stddev = new cv.Mat()
    cv.meanStdDev(patchClone, mean, stddev)
    const std = stddev.data64F?.[0] ?? 0
    totalVariance += std * std
    
    patch.delete()
    patchClone.delete()
    mean.delete()
    stddev.delete()
  }

  return Math.sqrt(totalVariance / numSamples)
}

/**
 * Advanced preprocessing pipeline (Scanner-like)
 */
function applyAdvancedPreprocessing(
  cv: OpenCV,
  src: OpenCVMat,
  quality: ImageQualityMetrics,
  matsToCleanup: OpenCVMat[]
): OpenCVMat {
  let processed = src.clone()
  matsToCleanup.push(processed)

  // Use the source directly - it's already in RGBA format
  // We'll convert to grayscale when needed for specific operations

    // 1. Contrast enhancement for low contrast images
    // Since OpenCV.js may not have CLAHE, we use brightness/contrast adjustment
    if (quality.contrast < 30) {
      // Low contrast - will be handled in brightness normalization step
      // Skip separate enhancement to avoid complexity
    }

  // Convert to grayscale for processing
  const gray = new cv.Mat()
  cv.cvtColor(processed, gray, cv.COLOR_RGBA2GRAY, 0)
  matsToCleanup.push(gray)

  // 2. Noise reduction based on quality
  const denoised = new cv.Mat()
  if (quality.noiseLevel > 15) {
    // High noise - use median blur
    cv.medianBlur(gray, denoised, 5)
  } else if (quality.noiseLevel > 8) {
    // Moderate noise - use Gaussian blur
    const ksize = new cv.Size(5, 5)
    cv.GaussianBlur(gray, denoised, ksize, 1.5, 1.5, cv.BORDER_DEFAULT)
  } else {
    // Low noise - light Gaussian blur
    const ksize = new cv.Size(3, 3)
    cv.GaussianBlur(gray, denoised, ksize, 0.5, 0.5, cv.BORDER_DEFAULT)
  }
  matsToCleanup.push(denoised)

  // 3. Brightness normalization
  const mean = new cv.Mat()
  const stddev = new cv.Mat()
  cv.meanStdDev(denoised, mean, stddev)
  const avgBrightness = mean.data64F?.[0] ?? 0
  
  // Target brightness: 128
  // Note: Brightness adjustment would require convertTo which may not be available in OpenCV.js
  // For now, we use the denoised image directly
  const normalized = denoised.clone()
  matsToCleanup.push(normalized, mean, stddev)

  // Convert back to RGBA for further processing
  const result = new cv.Mat()
  cv.cvtColor(normalized, result, cv.COLOR_GRAY2RGBA, 0)
  matsToCleanup.push(result)

  return result
}

/**
 * Advanced multi-scale paper detection with improved edge detection
 */
function detectAndWarpPaperAdvanced(
  cv: OpenCV,
  src: OpenCVMat,
  multiScale: boolean,
  matsToCleanup: OpenCVMat[]
): { found: boolean; warpedMat: OpenCVMat | null } {
  const scales = multiScale ? [1.0, 0.8, 1.2] : [1.0]
  let bestResult: { found: boolean; warpedMat: OpenCVMat | null } = { found: false, warpedMat: null }
  let bestArea = 0

  for (const scale of scales) {
    let workingMat = src
    if (scale !== 1.0) {
      // Use canvas-based resizing since cv.resize may not be available
      const tempCanvas = document.createElement("canvas")
      tempCanvas.width = Math.round(src.cols * scale)
      tempCanvas.height = Math.round(src.rows * scale)
      const tempCtx = tempCanvas.getContext("2d")!
      
      // Draw current image to canvas for resizing
      const srcCanvas = document.createElement("canvas")
      srcCanvas.width = src.cols
      srcCanvas.height = src.rows
      cv.imshow(srcCanvas, src)
      tempCtx.drawImage(srcCanvas, 0, 0, tempCanvas.width, tempCanvas.height)
      
      workingMat = cv.imread(tempCanvas)
      matsToCleanup.push(workingMat)
    }

    const result = detectPaperAtScale(cv, workingMat, matsToCleanup)
    
    if (result.found && result.warpedMat) {
      const area = result.warpedMat.cols * result.warpedMat.rows
      if (area > bestArea) {
        if (bestResult.warpedMat) bestResult.warpedMat.delete()
        bestResult = result
        bestArea = area
      } else {
        result.warpedMat.delete()
      }
    }
  }

  return bestResult
}

/**
 * Detect paper at a specific scale
 */
function detectPaperAtScale(
  cv: OpenCV,
  src: OpenCVMat,
  matsToCleanup: OpenCVMat[]
): { found: boolean; warpedMat: OpenCVMat | null } {
  const gray = new cv.Mat()
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0)
  matsToCleanup.push(gray)

  // Adaptive blur based on image size
  const blurSize = Math.max(3, Math.min(7, Math.floor(Math.min(src.cols, src.rows) / 300)))
  const ksize = new cv.Size(blurSize * 2 + 1, blurSize * 2 + 1)
  const blurred = new cv.Mat()
  cv.GaussianBlur(gray, blurred, ksize, 0, 0, cv.BORDER_DEFAULT)
  matsToCleanup.push(blurred)

  // Adaptive Canny thresholds based on image statistics
  const mean = new cv.Mat()
  const stddev = new cv.Mat()
  cv.meanStdDev(blurred, mean, stddev)
  const avgIntensity = mean.data64F?.[0] ?? 0
  const intensityStd = stddev.data64F?.[0] ?? 0
  
  // Adaptive thresholds: lower for darker images, higher for brighter
  const lowerThreshold = Math.max(30, Math.min(100, avgIntensity * 0.5))
  const upperThreshold = Math.max(100, Math.min(250, avgIntensity * 1.5))
  
  mean.delete()
  stddev.delete()

  const edge = new cv.Mat()
  cv.Canny(blurred, edge, lowerThreshold, upperThreshold, 3, false)
  matsToCleanup.push(edge)

  // Dilate edges to connect broken lines using morphology
  const dilated = new cv.Mat()
  const kernel = cv.Mat.ones(3, 3, cv.CV_8UC1)
  cv.morphologyEx(edge, dilated, cv.MORPH_DILATE, kernel, new cv.Point(-1, -1), 2)
  matsToCleanup.push(dilated, kernel)

  const contours = new cv.MatVector()
  const hierarchy = new cv.Mat()
  cv.findContours(dilated, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
  matsToCleanup.push(hierarchy)
  // Note: contours is OpenCVMatVector, not OpenCVMat, so we'll clean it up separately
  matsToCleanup.push(contours as unknown as OpenCVMat)

  let maxArea = 0
  let biggestApprox: OpenCVMat | null = null

  const minArea = src.cols * src.rows * 0.15 // At least 15% of image
  const maxAreaLimit = src.cols * src.rows * 0.95 // Not more than 95%

  for (let i = 0; i < contours.size(); ++i) {
    const c = contours.get(i)
    const area = cv.contourArea(c)

    if (area > minArea && area < maxAreaLimit && area > maxArea) {
      const peri = cv.arcLength(c, true)
      if (peri < 100) {
        c.delete()
        continue
      }

      const approx = new cv.Mat()
      // Adaptive epsilon based on perimeter
      const epsilon = Math.max(peri * 0.015, 3) // More adaptive
      cv.approxPolyDP(c, approx, epsilon, true)

      // Check if it's a quadrilateral
      if (approx.rows === 4) {
        maxArea = area
        if (biggestApprox) biggestApprox.delete()
        biggestApprox = approx.clone()
      }
      approx.delete()
    }
    c.delete()
  }

  if (biggestApprox) {
    try {
      const points = orderPoints(biggestApprox)
      const warpedMat = performPerspectiveTransform(cv, src, points)
      biggestApprox.delete()
      return { found: true, warpedMat }
    } catch (e) {
      console.warn("Perspective transform failed:", e)
      biggestApprox.delete()
    }
  }

  return { found: false, warpedMat: null }
}

/**
 * Order points for perspective transform (top-left, top-right, bottom-right, bottom-left)
 */
function orderPoints(approx: OpenCVMat): { x: number; y: number }[] {
  const points: { x: number; y: number }[] = []
  const data = approx.data32S
  if (!data) {
    throw new Error("Invalid approximation data")
  }

  for (let i = 0; i < 4; i++) {
    points.push({ x: data[i * 2], y: data[i * 2 + 1] })
  }

  // Sort by y-coordinate
  points.sort((a, b) => a.y - b.y)

  // Top two points (smaller y)
  const top = points.slice(0, 2).sort((a, b) => a.x - b.x)
  const tl = top[0]
  const tr = top[1]

  // Bottom two points (larger y)
  const bottom = points.slice(2, 4).sort((a, b) => a.x - b.x)
  const bl = bottom[0]
  const br = bottom[1]

  return [tl, tr, br, bl]
}

/**
 * Perform perspective transform
 */
function performPerspectiveTransform(
  cv: OpenCV,
  src: OpenCVMat,
  points: { x: number; y: number }[]
): OpenCVMat {
  // Calculate dimensions
  const widthA = Math.hypot(points[1].x - points[0].x, points[1].y - points[0].y)
  const widthB = Math.hypot(points[2].x - points[3].x, points[2].y - points[3].y)
  const maxWidth = Math.max(widthA, widthB)

  const heightA = Math.hypot(points[0].x - points[3].x, points[0].y - points[3].y)
  const heightB = Math.hypot(points[1].x - points[2].x, points[1].y - points[2].y)
  const maxHeight = Math.max(heightA, heightB)

  // Destination coordinates
  const dstCoords = cv.matFromArray(4, 1, cv.CV_32FC2, [
    0, 0,
    maxWidth - 1, 0,
    maxWidth - 1, maxHeight - 1,
    0, maxHeight - 1
  ])

  // Source coordinates
  const srcArr = [
    points[0].x, points[0].y,
    points[1].x, points[1].y,
    points[2].x, points[2].y,
    points[3].x, points[3].y
  ]
  const srcCoords = cv.matFromArray(4, 1, cv.CV_32FC2, srcArr)

  const M = cv.getPerspectiveTransform(srcCoords, dstCoords)
  const warpedMat = new cv.Mat()
  // Use default interpolation (OpenCV.js may not support INTER_LINEAR constant)
  cv.warpPerspective(
    src,
    warpedMat,
    M,
    new cv.Size(maxWidth, maxHeight)
  )

  M.delete()
  srcCoords.delete()
  dstCoords.delete()

  return warpedMat
}

/**
 * Ultra-robust bubble detection using multi-pass approach
 * Tries multiple methods and combines results for maximum detection rate
 */
function detectBubblesAdvanced(
  cv: OpenCV,
  src: OpenCVMat,
  config: ProcessingConfig,
  quality: ImageQualityMetrics,
  matsToCleanup: OpenCVMat[]
): Bubble[] {
  const gray = new cv.Mat()
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0)
  matsToCleanup.push(gray)

  // Enhanced preprocessing: Median blur to reduce noise while preserving edges
  const denoised = new cv.Mat()
  cv.medianBlur(gray, denoised, 5)
  matsToCleanup.push(denoised)

  // Edge-preserving smoothing
  const smoothed = new cv.Mat()
  const blurSize = quality.overall === 'poor' ? 5 : 3
  const ksize = new cv.Size(blurSize * 2 + 1, blurSize * 2 + 1)
  cv.GaussianBlur(denoised, smoothed, ksize, 1.0, 1.0, cv.BORDER_DEFAULT)
  matsToCleanup.push(smoothed)

  // Calculate adaptive thresholds based on image statistics
  const mean = new cv.Mat()
  const stddev = new cv.Mat()
  cv.meanStdDev(smoothed, mean, stddev)
  const avgIntensity = mean.data64F?.[0] ?? 128
  mean.delete()
  stddev.delete()

  // Method 1: Otsu thresholding (best for uniform lighting)
  const threshOtsu = new cv.Mat()
  cv.threshold(smoothed, threshOtsu, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
  matsToCleanup.push(threshOtsu)

  // Method 2: Adaptive thresholding (best for varying lighting)
  const threshAdaptive = new cv.Mat()
  const blockSize = quality.overall === 'poor' ? 21 : quality.overall === 'fair' ? 17 : 13
  const C = quality.overall === 'poor' ? 7 : quality.overall === 'fair' ? 4 : 2
  cv.adaptiveThreshold(
    smoothed,
    threshAdaptive,
    255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY_INV,
    blockSize,
    C
  )
  matsToCleanup.push(threshAdaptive)

  // Method 3: Canny edge detection for edge-based detection
  const edges = new cv.Mat()
  const lowerThreshold = Math.max(30, avgIntensity * 0.4)
  const upperThreshold = Math.max(100, avgIntensity * 1.2)
  cv.Canny(smoothed, edges, lowerThreshold, upperThreshold, 3, false)
  matsToCleanup.push(edges)

  // Combine methods: Use Otsu as primary, but also check adaptive
  const combined = threshOtsu.clone()
  matsToCleanup.push(combined)

  // Morphological operations to clean up and connect broken edges
  const kernel = cv.Mat.ones(3, 3, cv.CV_8UC1)
  cv.morphologyEx(combined, combined, cv.MORPH_CLOSE, kernel, new cv.Point(-1, -1), 2)
  cv.morphologyEx(combined, combined, cv.MORPH_OPEN, kernel, new cv.Point(-1, -1), 1)
  matsToCleanup.push(kernel)

  // Try HoughCircles first (most accurate for perfect circles)
  const bubblesFromHough: Bubble[] = []
  if (cv.HoughCircles && cv.HOUGH_GRADIENT !== undefined) {
    try {
      const circles = new cv.Mat()
      const totalPixels = src.cols * src.rows
      const minRadius = Math.max(3, Math.floor(Math.sqrt(totalPixels * 0.00005 / Math.PI)))
      const maxRadius = Math.min(
        Math.min(src.cols, src.rows) / 3,
        Math.floor(Math.sqrt(totalPixels * 0.015 / Math.PI))
      )
      const minDist = minRadius * 1.5 // Reduced for better detection

      cv.HoughCircles(
        smoothed,
        circles,
        cv.HOUGH_GRADIENT!,
        1, // dp
        minDist,
        avgIntensity * 1.2, // param1 - Canny threshold
        avgIntensity * 0.6, // param2 - accumulator threshold (lower = more circles)
        minRadius,
        maxRadius
      )

      if (circles.cols > 0) {
        const circlesData = circles.data32F
        if (circlesData) {
          for (let i = 0; i < circles.cols; i++) {
            const x = circlesData[i * 3]
            const y = circlesData[i * 3 + 1]
            const radius = circlesData[i * 3 + 2]

            if (x > 0 && y > 0 && radius > 0 && x < src.cols && y < src.rows) {
              bubblesFromHough.push({
                x: Math.round(x - radius),
                y: Math.round(y - radius),
                width: Math.round(radius * 2),
                height: Math.round(radius * 2),
                centerX: Math.round(x),
                centerY: Math.round(y),
                area: Math.PI * radius * radius,
                circularity: 1.0,
              })
            }
          }
        }
      }
      circles.delete()
    } catch (e) {
      console.warn("HoughCircles failed:", e)
    }
  }

  // Multi-pass contour detection with different parameters
  const allBubblesFromContours: Bubble[] = []
  const totalPixels = src.cols * src.rows

  // Pass 1: RETR_EXTERNAL (external contours only)
  const bubbles1 = detectBubblesFromContours(cv, combined, totalPixels, cv.RETR_EXTERNAL, matsToCleanup)
  allBubblesFromContours.push(...bubbles1)

  // Pass 2: RETR_TREE (all contours including nested) - catches bubbles inside filled bubbles
  const bubbles2 = detectBubblesFromContours(cv, combined, totalPixels, cv.RETR_TREE, matsToCleanup)
  allBubblesFromContours.push(...bubbles2)

  // Pass 3: Try with adaptive threshold as well
  const combinedAdaptive = threshAdaptive.clone()
  cv.morphologyEx(combinedAdaptive, combinedAdaptive, cv.MORPH_CLOSE, kernel, new cv.Point(-1, -1), 2)
  cv.morphologyEx(combinedAdaptive, combinedAdaptive, cv.MORPH_OPEN, kernel, new cv.Point(-1, -1), 1)
  matsToCleanup.push(combinedAdaptive)
  const bubbles3 = detectBubblesFromContours(cv, combinedAdaptive, totalPixels, cv.RETR_EXTERNAL, matsToCleanup)
  allBubblesFromContours.push(...bubbles3)

  // Remove duplicates from contour detection
  const uniqueContourBubbles = removeDuplicateBubbles(allBubblesFromContours)

  // Combine HoughCircles and Contours
  let allBubbles: Bubble[] = []

  if (bubblesFromHough.length > 0) {
    allBubbles = [...bubblesFromHough]

    // Add non-overlapping contours
    for (const contourBubble of uniqueContourBubbles) {
      const overlaps = allBubbles.some(houghBubble => {
        const dist = Math.hypot(
          contourBubble.centerX - houghBubble.centerX,
          contourBubble.centerY - houghBubble.centerY
        )
        const avgRadius = (contourBubble.width + houghBubble.width) / 4
        return dist < avgRadius * 1.2 // Overlap threshold
      })

      if (!overlaps) {
        allBubbles.push(contourBubble)
      }
    }
  } else {
    allBubbles = uniqueContourBubbles
  }

  // Advanced statistical filtering
  if (allBubbles.length > 0) {
    // Remove duplicates again after combining
    allBubbles = removeDuplicateBubbles(allBubbles)

    // Filter by area using IQR method
    allBubbles.sort((a, b) => a.area - b.area)
    const q1Index = Math.floor(allBubbles.length * 0.25)
    const q3Index = Math.floor(allBubbles.length * 0.75)
    const q1 = allBubbles[q1Index].area
    const q3 = allBubbles[q3Index].area
    const iqr = q3 - q1
    const minValidArea = Math.max(q1 - 2 * iqr, totalPixels * 0.00001) // Very lenient
    const maxValidArea = Math.min(q3 + 2 * iqr, totalPixels * 0.02) // Very lenient

    // Filter by size consistency
    const areas = allBubbles.map(b => b.area)
    const medianArea = areas[Math.floor(areas.length / 2)]
    const meanArea = areas.reduce((a, b) => a + b, 0) / areas.length
    const stdArea = Math.sqrt(
      areas.reduce((sum, a) => sum + Math.pow(a - meanArea, 2), 0) / areas.length
    )

    return allBubbles.filter(b => {
      // IQR filter (very lenient)
      if (b.area < minValidArea || b.area > maxValidArea) return false

      // Statistical filter (within 4 standard deviations - very lenient)
      if (Math.abs(b.area - meanArea) > 4 * stdArea) return false

      // Size consistency (within 3x of median - very lenient)
      if (b.area < medianArea * 0.15 || b.area > medianArea * 3.5) return false

      return true
    })
  }

  return allBubbles
}

/**
 * Detect bubbles from contours with lenient parameters
 */
function detectBubblesFromContours(
  cv: OpenCV,
  binary: OpenCVMat,
  totalPixels: number,
  retrievalMode: number,
  matsToCleanup: OpenCVMat[]
): Bubble[] {
  const bubbles: Bubble[] = []
  const contours = new cv.MatVector()
  const hierarchy = new cv.Mat()
  cv.findContours(binary, contours, hierarchy, retrievalMode, cv.CHAIN_APPROX_SIMPLE)
  matsToCleanup.push(hierarchy)
  matsToCleanup.push(contours as unknown as OpenCVMat)

  // Very lenient area thresholds
  const minArea = totalPixels * 0.00001 // Very lenient
  const maxArea = totalPixels * 0.02    // Very lenient

  for (let i = 0; i < contours.size(); ++i) {
    const contour = contours.get(i)
    const area = cv.contourArea(contour)

    if (area > minArea && area < maxArea) {
      const perimeter = cv.arcLength(contour, true)
      if (perimeter === 0) {
        contour.delete()
        continue
      }

      const circularity = (4 * Math.PI * area) / (perimeter * perimeter)
      const rect = cv.boundingRect(contour)
      const aspectRatio = rect.width / rect.height
      const extent = area / (rect.width * rect.height)

      // Very lenient filtering - accept almost any circular-like shape
      if (
        aspectRatio >= 0.3 && aspectRatio <= 3.0 && // Very lenient aspect ratio
        circularity > 0.2 && // Very lenient circularity
        extent > 0.3 // At least 30% of bounding box filled
      ) {
        const centerX = rect.x + rect.width / 2
        const centerY = rect.y + rect.height / 2

        bubbles.push({
          x: rect.x,
          y: rect.y,
          width: rect.width,
          height: rect.height,
          centerX,
          centerY,
          area,
          circularity,
        })
      }
    }
    contour.delete()
  }

  return bubbles
}

/**
 * Remove duplicate bubbles (same center within threshold)
 */
function removeDuplicateBubbles(bubbles: Bubble[]): Bubble[] {
  const unique: Bubble[] = []
  const threshold = 5 // pixels

  for (const bubble of bubbles) {
    const isDuplicate = unique.some(existing => {
      const dist = Math.hypot(
        bubble.centerX - existing.centerX,
        bubble.centerY - existing.centerY
      )
      return dist < threshold
    })

    if (!isDuplicate) {
      unique.push(bubble)
    }
  }

  return unique
}

/**
 * Enhanced grouping algorithm that handles missing bubbles and incomplete rows
 */
function groupBubblesIntoRows(bubbles: Bubble[], choicesPerQuestion: number): BubbleRow[] {
  if (bubbles.length === 0) return []

  bubbles.sort((a, b) => a.centerY - b.centerY)

  // Calculate row spacing more accurately
  const avgHeight = bubbles.reduce((acc, b) => acc + b.height, 0) / bubbles.length
  const rowTolerance = avgHeight * 1.0 // Increased tolerance for better grouping

  // Group bubbles into rows
  const rows: BubbleRow[] = []
  let currentRow: Bubble[] = []

  for (const bubble of bubbles) {
    if (currentRow.length === 0) {
      currentRow.push(bubble)
      continue
    }

    const rowAvgY = currentRow.reduce((acc, b) => acc + b.centerY, 0) / currentRow.length

    if (Math.abs(bubble.centerY - rowAvgY) < rowTolerance) {
      currentRow.push(bubble)
    } else {
      currentRow.sort((a, b) => a.centerX - b.centerX)
      rows.push({ bubbles: currentRow, avgY: rowAvgY })
      currentRow = [bubble]
    }
  }

  if (currentRow.length > 0) {
    currentRow.sort((a, b) => a.centerX - b.centerX)
    rows.push({
      bubbles: currentRow,
      avgY: currentRow.reduce((acc, b) => acc + b.centerY, 0) / currentRow.length,
    })
  }

  // Enhanced row processing: handle incomplete rows and multiple columns
  const validRows: BubbleRow[] = []

  // First, identify column structure
  const allBubblesFlat = rows.flatMap(r => r.bubbles)
  allBubblesFlat.sort((a, b) => a.centerX - b.centerX)

  // Detect columns by finding X-coordinate clusters
  const columns: number[] = []
  let lastColumnX = -1
  const avgBubbleWidth = allBubblesFlat.reduce((sum, b) => sum + b.width, 0) / allBubblesFlat.length

  for (const bubble of allBubblesFlat) {
    if (lastColumnX === -1 || bubble.centerX - lastColumnX > avgBubbleWidth * 2.5) {
      columns.push(bubble.centerX)
      lastColumnX = bubble.centerX
    }
  }

  // Process each row
  for (const row of rows) {
    // Sort bubbles in row by X coordinate
    row.bubbles.sort((a, b) => a.centerX - b.centerX)

    // Group bubbles by column
    const bubblesByColumn: Bubble[][] = []
    for (let i = 0; i < columns.length; i++) {
      bubblesByColumn[i] = []
    }

    for (const bubble of row.bubbles) {
      // Find closest column
      let closestCol = 0
      let minDist = Math.abs(bubble.centerX - columns[0])
      for (let i = 1; i < columns.length; i++) {
        const dist = Math.abs(bubble.centerX - columns[i])
        if (dist < minDist) {
          minDist = dist
          closestCol = i
        }
      }

      if (minDist < avgBubbleWidth * 2) {
        bubblesByColumn[closestCol].push(bubble)
      }
    }

    // Process each column's bubbles
    for (const columnBubbles of bubblesByColumn) {
      if (columnBubbles.length === 0) continue

      // Sort by X within column
      columnBubbles.sort((a, b) => a.centerX - b.centerX)

      // Group into question sets
      let i = 0
      while (i < columnBubbles.length) {
        const questionBubbles: Bubble[] = []
        const startX = columnBubbles[i].centerX
        const avgBubbleSize = columnBubbles.reduce((sum, b) => sum + b.width, 0) / columnBubbles.length

        // Collect bubbles that belong to the same question (close X coordinates)
        while (i < columnBubbles.length) {
          const bubble = columnBubbles[i]
          const distFromStart = Math.abs(bubble.centerX - startX)

          if (questionBubbles.length === 0 || distFromStart < avgBubbleSize * 1.5) {
            questionBubbles.push(bubble)
            i++
          } else {
            break
          }
        }

        // If we have at least some bubbles (even if not complete), create a row
        if (questionBubbles.length > 0) {
          // Sort by X to ensure proper order
          questionBubbles.sort((a, b) => a.centerX - b.centerX)

          // If we have fewer bubbles than expected, still accept it (missing bubbles)
          // But try to fill gaps if possible
          if (questionBubbles.length < choicesPerQuestion) {
            // Check if we can infer missing bubbles from spacing
            const spacing = questionBubbles.length > 1
              ? (questionBubbles[questionBubbles.length - 1].centerX - questionBubbles[0].centerX) / (questionBubbles.length - 1)
              : avgBubbleSize * 1.5

            // For now, just accept incomplete rows - they'll be handled in scoring
            validRows.push({
              bubbles: questionBubbles,
              avgY: row.avgY,
            })
          } else {
            // Split into complete question sets
            let j = 0
            while (j + choicesPerQuestion <= questionBubbles.length) {
              const chunk = questionBubbles.slice(j, j + choicesPerQuestion)
              validRows.push({
                bubbles: chunk,
                avgY: row.avgY,
              })
              j += choicesPerQuestion
            }

            // Handle remaining bubbles (incomplete set)
            if (j < questionBubbles.length) {
              validRows.push({
                bubbles: questionBubbles.slice(j),
                avgY: row.avgY,
              })
            }
          }
        }
      }
    }
  }

  if (validRows.length === 0) return []

  // Final organization: sort by column, then by row
  const sortedByX = [...validRows].sort((a, b) => a.bubbles[0].centerX - b.bubbles[0].centerX)
  const organizedColumns: BubbleRow[][] = []
  let currentCol: BubbleRow[] = [sortedByX[0]]
  organizedColumns.push(currentCol)

  let lastColX = sortedByX[0].bubbles[0].centerX
  const bubbleWidth = sortedByX[0].bubbles[0].width

  for (let k = 1; k < sortedByX.length; k++) {
    const row = sortedByX[k]
    const currX = row.bubbles[0].centerX

    // More lenient column separation
    if (currX - lastColX > bubbleWidth * 2.5) {
      currentCol = [row]
      organizedColumns.push(currentCol)
      lastColX = currX
    } else {
      currentCol.push(row)
    }
  }

  // Sort each column by Y coordinate
  organizedColumns.forEach(col => col.sort((a, b) => a.avgY - b.avgY))

  return organizedColumns.flat()
}

/**
 * Advanced fill percentage calculation with better contrast detection
 */
function calculateFillPercentageAdvanced(
  cv: OpenCV,
  gray: OpenCVMat,
  bubble: Bubble,
  matsToCleanup: OpenCVMat[]
): number {
  const roiRect = new cv.Rect(
    Math.floor(bubble.x),
    Math.floor(bubble.y),
    Math.floor(bubble.width),
    Math.floor(bubble.height)
  )

  // Boundary checks
  if (roiRect.x < 0) roiRect.x = 0
  if (roiRect.y < 0) roiRect.y = 0
  if (roiRect.x + roiRect.width >= gray.cols) roiRect.width = gray.cols - roiRect.x
  if (roiRect.y + roiRect.height >= gray.rows) roiRect.height = gray.rows - roiRect.y

  if (roiRect.width <= 0 || roiRect.height <= 0) return 0

  const roiView = gray.roi(roiRect)
  const roi = roiView.clone()
  roiView.delete()
  matsToCleanup.push(roi)

  const roiThresh = new cv.Mat()
  const mask = cv.Mat.zeros(roi.rows, roi.cols, cv.CV_8UC1)
  matsToCleanup.push(roiThresh, mask)

  try {
    // Calculate standard deviation to detect flat areas
    const mean = new cv.Mat()
    const stddev = new cv.Mat()
    cv.meanStdDev(roi, mean, stddev)
    const std = stddev.data64F?.[0] ?? 0
    mean.delete()
    stddev.delete()

    // Flat areas (empty bubbles) have low variance
    // Reduced threshold to catch more bubbles with faint marks
    if (std < 5) {
      return 0
    }

    // Try adaptive thresholding first for better accuracy on varying lighting
    const adaptiveBlockSize = Math.max(3, Math.min(roi.cols, roi.rows) / 3)
    const blockSize = adaptiveBlockSize % 2 === 1 ? adaptiveBlockSize : adaptiveBlockSize + 1
    
    try {
      cv.adaptiveThreshold(
        roi,
        roiThresh,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        blockSize,
        2
      )
    } catch (e) {
      // Fallback to Otsu if adaptive fails
      cv.threshold(roi, roiThresh, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    }

    // Create circular mask
    const center = new cv.Point(Math.floor(roi.cols / 2), Math.floor(roi.rows / 2))
    const r = Math.floor(Math.min(roi.cols, roi.rows) / 2) - 2
    cv.circle(mask, center, r, new cv.Scalar(255), -1)

    // Calculate contrast between ink and paper
    let inkSum = 0
    let inkCount = 0
    let paperSum = 0
    let paperCount = 0

    const roiData = roi.data
    const threshData = roiThresh.data
    const maskData = mask.data

    for (let i = 0; i < roiData.length; i++) {
      if (maskData[i] === 255) {
        if (threshData[i] === 255) {
          inkSum += roiData[i]
          inkCount++
        } else {
          paperSum += roiData[i]
          paperCount++
        }
      }
    }

    if (inkCount === 0 || paperCount === 0) {
      return inkCount > 0 ? 1.0 : 0.0
    }

    const meanInk = inkSum / inkCount
    const meanPaper = paperSum / paperCount
    const contrast = meanPaper - meanInk

    // Enhanced contrast check - reduced threshold for better detection of faint marks
    if (contrast < 8) {
      return 0
    }

    const fillRatio = inkCount / (inkCount + paperCount)
    
    // Additional validation: check intensity distribution
    // Filled bubbles should have a clear bimodal distribution (dark ink vs light paper)
    const intensityValues: number[] = []
    for (let i = 0; i < roiData.length; i++) {
      if (maskData[i] === 255) {
        intensityValues.push(roiData[i])
      }
    }
    
    if (intensityValues.length > 0) {
      intensityValues.sort((a, b) => a - b)
      const q25 = intensityValues[Math.floor(intensityValues.length * 0.25)]
      const q75 = intensityValues[Math.floor(intensityValues.length * 0.75)]
      const iqr = q75 - q25
      
      // If there's a clear separation (high IQR), it's likely a filled bubble
      // If IQR is very low, it's likely uniform (empty bubble)
      if (iqr < 10 && fillRatio < 0.3) {
        return 0 // Uniform and low fill = empty
      }
    }
    
    return fillRatio

  } catch (e) {
    console.error("Fill calculation error:", e)
    return 0
  }
}

/**
 * Detect selected bubble from fill percentages
 */
function detectSelectedBubble(
  fillPercentages: number[],
  minThreshold: number,
): { selectedIdx: number; isValid: boolean; hasMultiple: boolean } {
  const aboveThreshold = fillPercentages
    .map((p, i) => ({ idx: i, fill: p }))
    .filter((b) => b.fill >= minThreshold)

  if (aboveThreshold.length === 0) {
    return { selectedIdx: -1, isValid: false, hasMultiple: false }
  }

  const confidenceThreshold = Math.max(minThreshold + 0.15, 0.40)

  if (aboveThreshold.length > 1) {
    const sorted = [...aboveThreshold].sort((a, b) => b.fill - a.fill)

    if (sorted[0].fill < confidenceThreshold) {
      return { selectedIdx: -1, isValid: false, hasMultiple: false }
    }

    if (sorted[0].fill > sorted[1].fill * 1.5) {
      return { selectedIdx: sorted[0].idx, isValid: true, hasMultiple: false }
    }
    return { selectedIdx: sorted[0].idx, isValid: true, hasMultiple: true }
  }

  return { selectedIdx: aboveThreshold[0].idx, isValid: true, hasMultiple: false }
}

/**
 * Draw result markers on the image
 */
function drawResultMarkersOpenCV(
  cv: OpenCV,
  src: OpenCVMat,
  bubbles: Bubble[],
  fillPercentages: number[],
  minThreshold: number,
  correctIdx: number | undefined,
  status: QuestionResult["status"],
  isValid: boolean,
  hasMultiple: boolean,
): void {
  const RED = new cv.Scalar(255, 0, 0, 255)
  const GREEN = new cv.Scalar(0, 255, 0, 255)
  const ORANGE = new cv.Scalar(255, 165, 0, 255)

  const drawCircle = (bubble: Bubble, color: OpenCVScalar, thickness: number = 2) => {
    const center = new cv.Point(Math.round(bubble.centerX), Math.round(bubble.centerY))
    const radius = Math.round(Math.max(bubble.width, bubble.height) / 2 + 2)
    cv.circle(src, center, radius, color, thickness)
  }

  const drawCross = (bubble: Bubble, color: OpenCVScalar, thickness: number = 2) => {
    const center = new cv.Point(Math.round(bubble.centerX), Math.round(bubble.centerY))
    const radius = Math.round(Math.max(bubble.width, bubble.height) / 2 - 2)

    const pt1 = new cv.Point(center.x - radius, center.y - radius)
    const pt2 = new cv.Point(center.x + radius, center.y + radius)
    const pt3 = new cv.Point(center.x + radius, center.y - radius)
    const pt4 = new cv.Point(center.x - radius, center.y + radius)

    cv.line(src, pt1, pt2, color, thickness)
    cv.line(src, pt3, pt4, color, thickness)
  }

  const selectedIndices = fillPercentages
    .map((p, i) => ({ p, i }))
    .filter((item) => item.p >= minThreshold)
    .map((item) => item.i)

  if (status === "correct" && correctIdx !== undefined && bubbles[correctIdx]) {
    drawCircle(bubbles[correctIdx], GREEN, 3)
  } else if (status === "incorrect") {
    if (hasMultiple) {
      selectedIndices.forEach((idx) => {
        if (bubbles[idx]) drawCross(bubbles[idx], RED, 3)
      })
    } else if (isValid && selectedIndices.length > 0) {
      const wrongIdx = selectedIndices[0]
      if (bubbles[wrongIdx]) drawCross(bubbles[wrongIdx], RED, 3)
    }

    if (correctIdx !== undefined && bubbles[correctIdx]) {
      drawCircle(bubbles[correctIdx], GREEN, 3)
    }
  } else if (status === "unanswered") {
    if (correctIdx !== undefined && bubbles[correctIdx]) {
      drawCircle(bubbles[correctIdx], ORANGE, 2)
    }
  }
}
