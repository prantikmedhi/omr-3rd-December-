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
  detectedChoicesPerQuestion?: number
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
  radius?: number
}

interface BubbleRow {
  bubbles: Bubble[]
  avgY: number
}

// Global variable access for OpenCV.js
declare const cv: any

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
        if (typeof cv === 'undefined') {
            reject(new Error("OpenCV.js is not loaded yet."))
            return
        }
        const result = processImageOpenCV(img, answerKey, choicesPerQuestion, minFillThreshold, binaryThreshold)
        resolve(result)
      } catch (err) {
        console.error(err)
        reject(err)
      }
    }

    img.onerror = () => {
      reject(new Error("Failed to load image"))
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
 * Enhanced image preprocessing with CLAHE, denoising, and adaptive enhancement
 */
function enhanceImage(gray: any): any {
    let enhanced = new cv.Mat()
    
    // Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
    try {
        const clahe = new cv.CLAHE(2.0, new cv.Size(8, 8))
        clahe.apply(gray, enhanced)
        clahe.delete()
    } catch (e) {
        // Fallback: Use regular histogram equalization if CLAHE is not available
        console.warn("CLAHE not available, using regular histogram equalization")
        cv.equalizeHist(gray, enhanced)
    }
    
    // Denoise using Non-local Means Denoising (if available) or fastNlMeansDenoising
    try {
        const denoised = new cv.Mat()
        cv.fastNlMeansDenoising(enhanced, denoised, 3, 7, 21)
        enhanced.delete()
        return denoised
    } catch (e) {
        // Fallback: Use bilateral filter for denoising
        try {
            const denoised = new cv.Mat()
            cv.bilateralFilter(enhanced, denoised, 9, 75, 75)
            enhanced.delete()
            return denoised
        } catch (e2) {
            // Final fallback: Just return enhanced image with Gaussian blur
            const blurred = new cv.Mat()
            const ksize = new cv.Size(5, 5)
            cv.GaussianBlur(enhanced, blurred, ksize, 0, 0, cv.BORDER_DEFAULT)
            enhanced.delete()
            return blurred
        }
    }
}

/**
 * Multi-method bubble detection combining contour detection and Hough Circle Transform
 */
function detectBubblesMultiMethod(
    gray: any,
    enhanced: any,
    totalPixels: number
): Bubble[] {
    const bubbles: Bubble[] = []
    const bubbleMap = new Map<string, Bubble>()
    
    // Method 1: Enhanced Contour Detection
    const bubbles1 = detectBubblesByContours(enhanced, totalPixels)
    bubbles1.forEach(b => {
        const key = `${Math.round(b.centerX)},${Math.round(b.centerY)}`
        bubbleMap.set(key, b)
    })
    
    // Method 2: Hough Circle Transform (for circular bubbles)
    const bubbles2 = detectBubblesByHoughCircles(gray, enhanced, totalPixels)
    bubbles2.forEach(b => {
        const key = `${Math.round(b.centerX)},${Math.round(b.centerY)}`
        // Only add if not already detected by contours (merge nearby bubbles)
        if (!bubbleMap.has(key)) {
            // Check if it's close to an existing bubble
            let isDuplicate = false
            for (const [existingKey, existingBubble] of bubbleMap.entries()) {
                const dist = Math.hypot(
                    b.centerX - existingBubble.centerX,
                    b.centerY - existingBubble.centerY
                )
                if (dist < Math.max(b.width, existingBubble.width) * 0.5) {
                    isDuplicate = true
                    break
                }
            }
            if (!isDuplicate) {
                bubbleMap.set(key, b)
            }
        }
    })
    
    return Array.from(bubbleMap.values())
}

/**
 * Detect bubbles using contour analysis with improved filtering
 */
function detectBubblesByContours(enhanced: any, totalPixels: number): Bubble[] {
    const bubbles: Bubble[] = []
    const thresh = new cv.Mat()
    const contours = new cv.MatVector()
    const hierarchy = new cv.Mat()
    
    try {
        // Multi-scale adaptive thresholding
        const blockSize = Math.max(11, Math.floor(Math.sqrt(totalPixels) / 20) * 2 + 1)
        cv.adaptiveThreshold(
            enhanced,
            thresh,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY_INV,
            blockSize,
            3
        )
        
        // Morphological operations to improve bubble detection
        const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3))
        cv.morphologyEx(thresh, thresh, cv.MORPH_CLOSE, kernel)
        cv.morphologyEx(thresh, thresh, cv.MORPH_OPEN, kernel)
        kernel.delete()
        
        // Find contours
        cv.findContours(thresh, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        const minArea = totalPixels * 0.00003 // More lenient for small bubbles
        const maxArea = totalPixels * 0.01    // More lenient for large bubbles
        
        for (let i = 0; i < contours.size(); ++i) {
            const contour = contours.get(i)
            const area = cv.contourArea(contour)
            
            if (area > minArea && area < maxArea) {
                const perimeter = cv.arcLength(contour, true)
                if (perimeter === 0) continue
                
                const circularity = 4 * Math.PI * area / (perimeter * perimeter)
                const rect = cv.boundingRect(contour)
                const aspectRatio = rect.width / rect.height
                
                // More relaxed thresholds for various image qualities
                if (aspectRatio >= 0.5 && aspectRatio <= 2.0 && circularity > 0.3) {
                    const centerX = rect.x + rect.width / 2
                    const centerY = rect.y + rect.height / 2
                    const radius = Math.sqrt(area / Math.PI)
                    
                    bubbles.push({
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height,
                        centerX,
                        centerY,
                        area,
                        circularity,
                        radius
                    })
                }
            }
            contour.delete()
        }
    } finally {
        thresh.delete()
        contours.delete()
        hierarchy.delete()
    }
    
    return bubbles
}

/**
 * Detect bubbles using Hough Circle Transform (alternative method)
 */
function detectBubblesByHoughCircles(
    gray: any,
    enhanced: any,
    totalPixels: number
): Bubble[] {
    const bubbles: Bubble[] = []
    const circles = new cv.Mat()
    
    try {
        // Estimate bubble radius range based on image size
        const minRadius = Math.floor(Math.sqrt(totalPixels) * 0.01)
        const maxRadius = Math.floor(Math.sqrt(totalPixels) * 0.1)
        
        if (minRadius < 5 || maxRadius < minRadius) {
            return bubbles // Skip if radius range is invalid
        }
        
        // Apply Hough Circle Transform
        cv.HoughCircles(
            enhanced,
            circles,
            cv.HOUGH_GRADIENT,
            1,                    // dp: inverse ratio of accumulator resolution
            minRadius * 2,        // minDist: minimum distance between centers
            100,                  // param1: upper threshold for edge detection
            30,                   // param2: accumulator threshold (lower = more circles)
            minRadius,            // minRadius
            maxRadius             // maxRadius
        )
        
        // Convert circles to Bubble format
        for (let i = 0; i < circles.cols; i++) {
            const x = Math.round(circles.data32F[i * 3])
            const y = Math.round(circles.data32F[i * 3 + 1])
            const r = Math.round(circles.data32F[i * 3 + 2])
            
            bubbles.push({
                x: x - r,
                y: y - r,
                width: r * 2,
                height: r * 2,
                centerX: x,
                centerY: y,
                area: Math.PI * r * r,
                circularity: 1.0, // Perfect circle
                radius: r
            })
        }
    } catch (e) {
        console.warn("Hough Circle detection failed:", e)
    } finally {
        circles.delete()
    }
    
    return bubbles
}

/**
 * Automatically detect the number of choices per question by analyzing bubble patterns
 */
function detectChoicesPerQuestion(bubbles: Bubble[]): number {
    if (bubbles.length === 0) return 4 // Default
    
    // Group bubbles by rows
    bubbles.sort((a, b) => a.centerY - b.centerY)
    const avgHeight = bubbles.reduce((acc, b) => acc + b.height, 0) / bubbles.length
    const rowTolerance = avgHeight * 0.6
    
    const rows: Bubble[][] = []
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
            rows.push(currentRow)
            currentRow = [bubble]
        }
    }
    if (currentRow.length > 0) {
        currentRow.sort((a, b) => a.centerX - b.centerX)
        rows.push(currentRow)
    }
    
    // Count bubble counts per row
    const counts = new Map<number, number>()
    rows.forEach(row => {
        const count = row.length
        counts.set(count, (counts.get(count) || 0) + 1)
    })
    
    // Find the most common count (likely choices per question)
    let maxCount = 0
    let mostCommon = 4 // Default
    
    for (const [count, frequency] of counts.entries()) {
        if (frequency > maxCount && count >= 2 && count <= 10) {
            maxCount = frequency
            mostCommon = count
        }
    }
    
    console.log(`[OMR] Auto-detected ${mostCommon} choices per question`)
    return mostCommon
}

/**
 * Enhanced perspective correction with better edge detection
 */
function detectAndWarpPaper(src: any): { found: boolean, warpedMat: any | null } {
    let gray = new cv.Mat()
    let blurred = new cv.Mat()
    let edge = new cv.Mat()
    let contours = new cv.MatVector()
    let hierarchy = new cv.Mat()
    let warpedMat = null
    let found = false

    try {
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0)

        // Enhanced blur for better edge detection
        const ksize = new cv.Size(5, 5)
        cv.GaussianBlur(gray, blurred, ksize, 0, 0, cv.BORDER_DEFAULT)

        // Improved Canny edge detection with adaptive thresholds
        const median = cv.mean(blurred).data[0]
        const lower = Math.max(0, 0.66 * median)
        const upper = Math.min(255, 1.33 * median)
        cv.Canny(blurred, edge, lower, upper)

        // Dilate edges to connect broken lines
        const dilateKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3))
        cv.dilate(edge, edge, dilateKernel)
        dilateKernel.delete()

        cv.findContours(edge, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        let maxArea = 0
        let biggestApprox = null

        for (let i = 0; i < contours.size(); ++i) {
            const c = contours.get(i)
            const area = cv.contourArea(c)

            if (area > (src.cols * src.rows * 0.1)) {
                const peri = cv.arcLength(c, true)
                const approx = new cv.Mat()
                // More lenient approximation for perspective correction
                cv.approxPolyDP(c, approx, 0.02 * peri, true)

                if (approx.rows === 4 && area > maxArea) {
                    maxArea = area
                    if (biggestApprox) biggestApprox.delete()
                    biggestApprox = approx.clone()
                }
                approx.delete()
            }
            c.delete()
        }

        if (biggestApprox) {
            found = true
            const points = orderPoints(biggestApprox)

            const widthA = Math.hypot(points[1].x - points[0].x, points[1].y - points[0].y)
            const widthB = Math.hypot(points[2].x - points[3].x, points[2].y - points[3].y)
            const maxWidth = Math.max(widthA, widthB)

            const heightA = Math.hypot(points[0].x - points[3].x, points[0].y - points[3].y)
            const heightB = Math.hypot(points[1].x - points[2].x, points[1].y - points[2].y)
            const maxHeight = Math.max(heightA, heightB)

            const dstCoords = cv.matFromArray(4, 1, cv.CV_32FC2, [
                0, 0,
                maxWidth - 1, 0,
                maxWidth - 1, maxHeight - 1,
                0, maxHeight - 1
            ])

            const srcArr = [
                points[0].x, points[0].y,
                points[1].x, points[1].y,
                points[2].x, points[2].y,
                points[3].x, points[3].y
            ]
            const srcCoords = cv.matFromArray(4, 1, cv.CV_32FC2, srcArr)

            const M = cv.getPerspectiveTransform(srcCoords, dstCoords)
            warpedMat = new cv.Mat()
            cv.warpPerspective(src, warpedMat, M, new cv.Size(maxWidth, maxHeight))

            M.delete()
            srcCoords.delete()
            dstCoords.delete()
            biggestApprox.delete()
        }

    } catch (e) {
        console.error("Warp failed", e)
        found = false
        if (warpedMat) warpedMat.delete()
        warpedMat = null
    } finally {
        gray.delete()
        blurred.delete()
        edge.delete()
        contours.delete()
        hierarchy.delete()
    }

    return { found, warpedMat }
}

function orderPoints(approx: any): {x: number, y: number}[] {
    const points = []
    const data = approx.data32S
    for (let i = 0; i < 4; i++) {
        points.push({ x: data[i * 2], y: data[i * 2 + 1] })
    }

    points.sort((a, b) => a.y - b.y)

    const top = points.slice(0, 2).sort((a, b) => a.x - b.x)
    const tl = top[0]
    const tr = top[1]

    const bottom = points.slice(2, 4).sort((a, b) => a.x - b.x)
    const bl = bottom[0]
    const br = bottom[1]

    return [tl, tr, br, bl]
}

/**
 * Enhanced bubble grouping with automatic grid detection
 */
function groupBubblesIntoRows(bubbles: Bubble[], choicesPerQuestion: number, autoDetect: boolean = true): BubbleRow[] {
    if (bubbles.length === 0) return []
    
    // Auto-detect choices per question if enabled
    let detectedChoices = choicesPerQuestion
    if (autoDetect) {
        detectedChoices = detectChoicesPerQuestion(bubbles)
    }
    
    bubbles.sort((a, b) => a.centerY - b.centerY)

    // Improved row tolerance calculation
    const avgHeight = bubbles.reduce((acc, b) => acc + b.height, 0) / bubbles.length
    const avgWidth = bubbles.reduce((acc, b) => acc + b.width, 0) / bubbles.length
    const rowTolerance = Math.max(avgHeight * 0.5, avgWidth * 0.3)

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
        rows.push({ bubbles: currentRow, avgY: currentRow.reduce((acc, b) => acc + b.centerY, 0) / currentRow.length })
    }

    const validRows: BubbleRow[] = []

    // More flexible grouping - handle variable bubbles per row
    for (const row of rows) {
        if (row.bubbles.length >= detectedChoices) {
            // Try to group into questions
            let i = 0
            while (i + detectedChoices <= row.bubbles.length) {
                const chunk = row.bubbles.slice(i, i + detectedChoices)
                validRows.push({
                    bubbles: chunk,
                    avgY: row.avgY
                })
                i += detectedChoices
            }
            // Handle remaining bubbles if they form a valid question
            const remaining = row.bubbles.slice(i)
            if (remaining.length >= Math.floor(detectedChoices * 0.7)) {
                // Allow partial questions (70% of expected bubbles)
                validRows.push({
                    bubbles: remaining,
                    avgY: row.avgY
                })
            }
        } else if (row.bubbles.length >= Math.floor(detectedChoices * 0.7)) {
            // Allow rows with fewer bubbles (might be edge cases)
            validRows.push({
                bubbles: row.bubbles,
                avgY: row.avgY
            })
        }
    }

    if (validRows.length === 0) return []

    // Group by columns (handle multi-column layouts)
    const sortedByX = [...validRows].sort((a,b) => a.bubbles[0].centerX - b.bubbles[0].centerX)

    const columns: BubbleRow[][] = []
    let currentCol: BubbleRow[] = [sortedByX[0]]
    columns.push(currentCol)

    let lastX = sortedByX[0].bubbles[0].centerX
    const avgBubbleWidth = sortedByX[0].bubbles.reduce((acc, b) => acc + b.width, 0) / sortedByX[0].bubbles.length

    for (let k = 1; k < sortedByX.length; k++) {
        const row = sortedByX[k]
        const currX = row.bubbles[0].centerX

        // More flexible column detection
        if (currX - lastX > avgBubbleWidth * 2.5) {
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
 * Enhanced fill percentage calculation with multiple methods
 */
function calculateFillPercentageOpenCV(gray: any, bubble: Bubble): number {
    const roiRect = new cv.Rect(
        Math.floor(bubble.x),
        Math.floor(bubble.y),
        Math.floor(bubble.width),
        Math.floor(bubble.height)
    )

    if (roiRect.x < 0) roiRect.x = 0
    if (roiRect.y < 0) roiRect.y = 0
    if (roiRect.x + roiRect.width >= gray.cols) roiRect.width = gray.cols - roiRect.x
    if (roiRect.y + roiRect.height >= gray.rows) roiRect.height = gray.rows - roiRect.y

    if (roiRect.width <= 0 || roiRect.height <= 0) return 0

    const roiView = gray.roi(roiRect)
    const roi = roiView.clone()
    roiView.delete()

    const roiThresh = new cv.Mat()
    const mask = new cv.Mat.zeros(roi.rows, roi.cols, cv.CV_8UC1)

    try {
        // Method 1: Standard deviation check for flatness
        const mean = new cv.Mat()
        const stddev = new cv.Mat()
        cv.meanStdDev(roi, mean, stddev)
        const std = stddev.data64F[0]
        const meanVal = mean.data64F[0]
        mean.delete()
        stddev.delete()

        // Adaptive threshold based on image quality
        const stdThreshold = meanVal < 100 ? 8 : 12 // Lower threshold for darker images
        
        if (std < stdThreshold) {
            return 0
        }

        // Method 2: Otsu's Thresholding
        const thresholdVal = cv.threshold(roi, roiThresh, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        // Create circular mask for more accurate detection
        const center = new cv.Point(Math.floor(roi.cols / 2), Math.floor(roi.rows / 2))
        const r = bubble.radius 
            ? Math.floor(bubble.radius * 0.9) // Use detected radius if available
            : Math.floor(Math.min(roi.cols, roi.rows) / 2 * 0.9)
        cv.circle(mask, center, r, new cv.Scalar(255), -1)
        
        const roiData = roi.data
        const threshData = roiThresh.data
        const maskData = mask.data

        let inkSum = 0
        let inkCount = 0
        let paperSum = 0
        let paperCount = 0

        for (let i = 0; i < roiData.length; i++) {
            if (maskData[i] === 255) {
                if (threshData[i] === 255) { // Ink
                    inkSum += roiData[i]
                    inkCount++
                } else { // Paper
                    paperSum += roiData[i]
                    paperCount++
                }
            }
        }

        if (inkCount === 0 || paperCount === 0) {
            // Check if it's uniformly dark (filled) or light (empty)
            return (meanVal < 100) ? 1.0 : 0.0
        }

        const meanInk = inkSum / inkCount
        const meanPaper = paperSum / paperCount
        const contrast = meanPaper - meanInk

        // Adaptive contrast threshold
        const contrastThreshold = meanVal < 100 ? 15 : 20
        
        if (contrast < contrastThreshold) {
            return 0
        }

        // Calculate fill percentage
        const fillRatio = inkCount / (inkCount + paperCount)
        
        // Boost confidence for high contrast fills
        if (contrast > 50 && fillRatio > 0.3) {
            return Math.min(1.0, fillRatio * 1.2)
        }
        
        return fillRatio

    } catch (e) {
        console.error("Fill Calc Error", e)
        return 0
    } finally {
        roi.delete()
        roiThresh.delete()
        mask.delete()
    }
}

function detectSelectedBubble(
  fillPercentages: number[],
  minThreshold: number,
): { selectedIdx: number; isValid: boolean; hasMultiple: boolean } {
  const aboveThreshold = fillPercentages.map((p, i) => ({ idx: i, fill: p })).filter((b) => b.fill >= minThreshold)

  if (aboveThreshold.length === 0) {
    return { selectedIdx: -1, isValid: false, hasMultiple: false }
  }

  const confidenceThreshold = Math.max(minThreshold + 0.15, 0.40)

  if (aboveThreshold.length > 1) {
    const sorted = [...aboveThreshold].sort((a, b) => b.fill - a.fill)

    if (sorted[0].fill < confidenceThreshold) {
       return { selectedIdx: -1, isValid: false, hasMultiple: false }
    }

    // More lenient ratio for multiple selections
    if (sorted[0].fill > sorted[1].fill * 1.4) {
      return { selectedIdx: sorted[0].idx, isValid: true, hasMultiple: false }
    }
    return { selectedIdx: sorted[0].idx, isValid: true, hasMultiple: true }
  }

  return { selectedIdx: aboveThreshold[0].idx, isValid: true, hasMultiple: false }
}

function drawResultMarkersOpenCV(
    src: any,
    bubbles: Bubble[],
    fillPercentages: number[],
    minThreshold: number,
    correctIdx: number | undefined,
    status: QuestionResult["status"],
    isValid: boolean,
    hasMultiple: boolean
): void {
    const RED = new cv.Scalar(255, 0, 0, 255)
    const GREEN = new cv.Scalar(0, 255, 0, 255)
    const ORANGE = new cv.Scalar(255, 165, 0, 255)

    const drawCircle = (bubble: Bubble, color: any, thickness: number = 2) => {
        const center = new cv.Point(Math.round(bubble.centerX), Math.round(bubble.centerY))
        const radius = bubble.radius 
            ? Math.round(bubble.radius + 2)
            : Math.round(Math.max(bubble.width, bubble.height) / 2 + 2)
        cv.circle(src, center, radius, color, thickness)
    }

    const drawCross = (bubble: Bubble, color: any, thickness: number = 2) => {
        const center = new cv.Point(Math.round(bubble.centerX), Math.round(bubble.centerY))
        const radius = bubble.radius 
            ? Math.round(bubble.radius - 2)
            : Math.round(Math.max(bubble.width, bubble.height) / 2 - 2)

        const pt1 = new cv.Point(center.x - radius, center.y - radius)
        const pt2 = new cv.Point(center.x + radius, center.y + radius)
        const pt3 = new cv.Point(center.x + radius, center.y - radius)
        const pt4 = new cv.Point(center.x - radius, center.y + radius)

        cv.line(src, pt1, pt2, color, thickness)
        cv.line(src, pt3, pt4, color, thickness)
    }

    const selectedIndices = fillPercentages
        .map((p, i) => ({ p, i }))
        .filter(item => item.p >= minThreshold)
        .map(item => item.i)

    if (status === "correct" && correctIdx !== undefined && bubbles[correctIdx]) {
        drawCircle(bubbles[correctIdx], GREEN, 3)
    } else if (status === "incorrect") {
        if (hasMultiple) {
            selectedIndices.forEach(idx => {
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

function processImageOpenCV(
  img: HTMLImageElement,
  answerKey: Record<number, number>,
  choicesPerQuestion: number,
  minFillThreshold: number,
  binaryThreshold: number,
): OMRResult {
    const canvas = document.createElement("canvas")
    const ctx = canvas.getContext("2d")!
    canvas.width = img.width
    canvas.height = img.height
    ctx.drawImage(img, 0, 0, img.width, img.height)

    // Smart resizing - maintain aspect ratio, optimize for processing
    const maxDim = 3000 // Increased for better quality
    if (img.width > maxDim || img.height > maxDim) {
        const scale = maxDim / Math.max(img.width, img.height)
        canvas.width = Math.round(img.width * scale)
        canvas.height = Math.round(img.height * scale)
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
    }

    let src = cv.imread(canvas)
    let warped = new cv.Mat()
    let gray = new cv.Mat()
    let enhanced = new cv.Mat()
    let hierarchy = new cv.Mat()
    let contours = new cv.MatVector()

    try {
        // 1. Enhanced Perspective Correction
        const { found, warpedMat } = detectAndWarpPaper(src)

        if (found && warpedMat) {
             src.delete()
             src = warpedMat.clone()
             warpedMat.delete()
        }

        // 2. Convert to grayscale
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0)

        // 3. Enhanced Image Preprocessing
        enhanced = enhanceImage(gray)

        // 4. Multi-method Bubble Detection
        const totalPixels = src.cols * src.rows
        const bubbles = detectBubblesMultiMethod(gray, enhanced, totalPixels)

        console.log(`[OMR] Found ${bubbles.length} potential bubbles using multi-method detection.`)

        // 5. Filter by Median Area (remove outliers)
        if (bubbles.length > 0) {
            bubbles.sort((a, b) => a.area - b.area)
            const medianArea = bubbles[Math.floor(bubbles.length / 2)].area
            const minValidArea = medianArea * 0.3  // More lenient
            const maxValidArea = medianArea * 2.0   // More lenient

            const filteredBubbles = bubbles.filter(b => b.area >= minValidArea && b.area <= maxValidArea)
            bubbles.length = 0
            bubbles.push(...filteredBubbles)
            
            console.log(`[OMR] Filtered to ${bubbles.length} bubbles after median area filtering.`)
        }

        if (bubbles.length === 0) {
             throw new Error("No bubbles detected. Please ensure the image is clear and bubbles are visible.")
        }

        // 6. Auto-detect choices per question if not provided or if 0
        let detectedChoices = choicesPerQuestion
        if (choicesPerQuestion <= 0 || choicesPerQuestion > 10) {
            detectedChoices = detectChoicesPerQuestion(bubbles)
        }

        // 7. Enhanced Bubble Grouping with auto-detection
        const rows = groupBubblesIntoRows(bubbles, detectedChoices, true)

        if (rows.length === 0) {
            throw new Error("Could not group bubbles into rows. Please check image quality and bubble visibility.")
        }

        console.log(`[OMR] Grouped into ${rows.length} question rows.`)

        // 8. Score Questions
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

            // Handle rows with fewer bubbles than expected
            if (!row.bubbles || row.bubbles.length === 0) {
                results.push({
                    question: questionNum,
                    status: "unanswered",
                    selected: null,
                    correctAnswer: correctAns ?? null,
                    fillPercentages: [],
                })
                unanswered++
                continue
            }

            const fillPercentages = row.bubbles.map(bubble => {
                return calculateFillPercentageOpenCV(enhanced, bubble)
            })

            const { selectedIdx, isValid, hasMultiple } = detectSelectedBubble(fillPercentages, minFillThreshold)

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

            drawResultMarkersOpenCV(src, row.bubbles, fillPercentages, minFillThreshold, correctAns, status, isValid, hasMultiple)
        }

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
            detectedChoicesPerQuestion: detectedChoices,
        }

    } finally {
        if (src) src.delete()
        if (warped) warped.delete()
        if (gray) gray.delete()
        if (enhanced) enhanced.delete()
        if (hierarchy) hierarchy.delete()
        if (contours) contours.delete()
    }
}
