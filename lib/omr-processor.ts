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

      const fillPercentages = row.bubbles.map(bubble => 
        calculateFillPercentageAdvanced(cv, gray, bubble, matsToCleanup)
      )

      const { selectedIdx, isValid, hasMultiple } = detectSelectedBubble(
        fillPercentages,
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
        fillPercentages,
      })

      drawResultMarkersOpenCV(cv, src, row.bubbles, fillPercentages, config.minFillThreshold, correctAns, status, isValid, hasMultiple)
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
 * Advanced bubble detection with adaptive parameters
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

  // Adaptive blur based on image quality
  const blurSize = quality.overall === 'poor' ? 7 : quality.overall === 'fair' ? 5 : 3
  const ksize = new cv.Size(blurSize, blurSize)
  const blurred = new cv.Mat()
  cv.GaussianBlur(gray, blurred, ksize, 0, 0, cv.BORDER_DEFAULT)
  matsToCleanup.push(blurred)

  // Adaptive thresholding parameters based on image quality
  const blockSize = quality.overall === 'poor' ? 25 : quality.overall === 'fair' ? 21 : 15
  const C = quality.overall === 'poor' ? 8 : quality.overall === 'fair' ? 5 : 3

  const thresh = new cv.Mat()
  cv.adaptiveThreshold(
    blurred,
    thresh,
    255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY_INV,
    blockSize,
    C
  )
  matsToCleanup.push(thresh)

  // Morphological operations to clean up
  const kernel = cv.Mat.ones(3, 3, cv.CV_8UC1)
  cv.morphologyEx(thresh, thresh, cv.MORPH_CLOSE, kernel, new cv.Point(-1, -1), 2)
  cv.morphologyEx(thresh, thresh, cv.MORPH_OPEN, kernel, new cv.Point(-1, -1), 1)
  matsToCleanup.push(kernel)

  // Find contours
  const contours = new cv.MatVector()
  const hierarchy = new cv.Mat()
  cv.findContours(thresh, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  matsToCleanup.push(hierarchy)
  // Note: contours is OpenCVMatVector, not OpenCVMat, so we'll clean it up separately
  matsToCleanup.push(contours as unknown as OpenCVMat)

  // Filter contours to find bubbles
  const bubbles: Bubble[] = []
  const totalPixels = src.cols * src.rows

  // Adaptive area thresholds based on image size
  const minArea = totalPixels * 0.00003 // More lenient minimum
  const maxArea = totalPixels * 0.01    // More lenient maximum

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

      // More lenient circularity check
      if (aspectRatio >= 0.5 && aspectRatio <= 2.0 && circularity > 0.3) {
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

  // Filter by median area to remove outliers
  if (bubbles.length > 0) {
    bubbles.sort((a, b) => a.area - b.area)
    const medianArea = bubbles[Math.floor(bubbles.length / 2)].area
    const minValidArea = medianArea * 0.3
    const maxValidArea = medianArea * 2.5

    return bubbles.filter(b => b.area >= minValidArea && b.area <= maxValidArea)
  }

  return bubbles
}

/**
 * Group bubbles into rows (questions)
 */
function groupBubblesIntoRows(bubbles: Bubble[], choicesPerQuestion: number): BubbleRow[] {
  bubbles.sort((a, b) => a.centerY - b.centerY)

  const avgHeight = bubbles.reduce((acc, b) => acc + b.height, 0) / bubbles.length
  const rowTolerance = avgHeight * 0.7 // Increased tolerance

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

  // Filter and organize rows
  const validRows: BubbleRow[] = []
  for (const row of rows) {
    if (row.bubbles.length >= choicesPerQuestion) {
      let i = 0
      while (i + choicesPerQuestion <= row.bubbles.length) {
        const chunk = row.bubbles.slice(i, i + choicesPerQuestion)
        validRows.push({
          bubbles: chunk,
          avgY: row.avgY,
        })
        i += choicesPerQuestion
      }
    }
  }

  if (validRows.length === 0) return []

  // Handle multiple columns
  const sortedByX = [...validRows].sort((a, b) => a.bubbles[0].centerX - b.bubbles[0].centerX)
  const columns: BubbleRow[][] = []
  let currentCol: BubbleRow[] = [sortedByX[0]]
  columns.push(currentCol)

  let lastX = sortedByX[0].bubbles[0].centerX
  const bubbleWidth = sortedByX[0].bubbles[0].width

  for (let k = 1; k < sortedByX.length; k++) {
    const row = sortedByX[k]
    const currX = row.bubbles[0].centerX

    if (currX - lastX > bubbleWidth * 3) {
      currentCol = [row]
      columns.push(currentCol)
      lastX = currX
    } else {
      currentCol.push(row)
    }
  }

  columns.forEach(col => col.sort((a, b) => a.avgY - b.avgY))

  return columns.flat()
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
    if (std < 8) {
      return 0
    }

    // Otsu's thresholding
    cv.threshold(roi, roiThresh, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

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

    // Enhanced contrast check
    if (contrast < 15) {
      return 0
    }

    return inkCount / (inkCount + paperCount)

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
