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

    // Resize for consistent processing if too large
    const maxDim = 2000
    if (img.width > maxDim || img.height > maxDim) {
        const scale = maxDim / Math.max(img.width, img.height)
        canvas.width = Math.round(img.width * scale)
        canvas.height = Math.round(img.height * scale)
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
    } else {
        // Just draw original
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
    }

    let src = cv.imread(canvas)
    let warped = new cv.Mat()
    let gray = new cv.Mat()
    let blurred = new cv.Mat()
    let thresh = new cv.Mat()
    let hierarchy = new cv.Mat()
    let contours = new cv.MatVector()

    try {
        // 1. Perspective Correction
        const { found, warpedMat } = detectAndWarpPaper(src)

        if (found && warpedMat) {
             src.delete()
             src = warpedMat.clone()
             warpedMat.delete()
        }

        // 2. Preprocessing (Adaptive Thresholding)
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0)

        // Gaussian Blur
        const ksize = new cv.Size(5, 5)
        cv.GaussianBlur(gray, blurred, ksize, 0, 0, cv.BORDER_DEFAULT)

        // Adaptive Thresholding for BUBBLE DETECTION only
        // Reduced C constant (10 -> 5) to better detect faint/broken bubbles
        cv.adaptiveThreshold(blurred, thresh, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 5)

        // Morphological Closing to connect broken contours
        const kernel = cv.Mat.ones(3, 3, cv.CV_8U)
        cv.morphologyEx(thresh, thresh, cv.MORPH_CLOSE, kernel)
        kernel.delete()

        // 3. Find Contours
        cv.findContours(thresh, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        // 4. Filter Contours (Find Bubbles)
        const bubbles: Bubble[] = []
        const totalPixels = src.cols * src.rows

        const minArea = totalPixels * 0.00005
        const maxArea = totalPixels * 0.005

        for (let i = 0; i < contours.size(); ++i) {
            const contour = contours.get(i)
            const area = cv.contourArea(contour)

            if (area > minArea && area < maxArea) {
                const perimeter = cv.arcLength(contour, true)
                if (perimeter === 0) continue
                const circularity = 4 * Math.PI * area / (perimeter * perimeter)

                const rect = cv.boundingRect(contour)
                const aspectRatio = rect.width / rect.height

                // Relaxed circularity threshold (0.6 -> 0.4) to support low quality / noisy images
                if (aspectRatio >= 0.6 && aspectRatio <= 1.5 && circularity > 0.4) {
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
                        circularity
                    })
                }
            }
        }

        console.log(`[OMR] Found ${bubbles.length} potential bubbles.`)

        // 5. Filter by Median Area
        if (bubbles.length > 0) {
            bubbles.sort((a, b) => a.area - b.area)
            const medianArea = bubbles[Math.floor(bubbles.length / 2)].area
            const minValidArea = medianArea * 0.4
            const maxValidArea = medianArea * 1.8

            const filteredBubbles = bubbles.filter(b => b.area >= minValidArea && b.area <= maxValidArea)
            bubbles.length = 0
            bubbles.push(...filteredBubbles)
        }

        if (bubbles.length === 0) {
             throw new Error("No bubbles detected. Adjust lighting or camera angle.")
        }

        // 6. Sort and Group Bubbles
        const rows = groupBubblesIntoRows(bubbles, choicesPerQuestion)

        // 7. Score
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

            const fillPercentages = row.bubbles.map(bubble => {
                return calculateFillPercentageOpenCV(gray, bubble)
            })

            const { selectedIdx, isValid, hasMultiple } = detectSelectedBubble(fillPercentages, minFillThreshold)

            let status: QuestionResult["status"]
            if (!isValid) {
                status = "unanswered"
                unanswered++
            } else if (hasMultiple) {
                // User requirement: Mark multiple as incorrect
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
        }

    } finally {
        if (src) src.delete()
        if (warped) warped.delete()
        if (gray) gray.delete()
        if (blurred) blurred.delete()
        if (thresh) thresh.delete()
        if (hierarchy) hierarchy.delete()
        if (contours) contours.delete()
    }
}

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

        const ksize = new cv.Size(5, 5)
        cv.GaussianBlur(gray, blurred, ksize, 0, 0, cv.BORDER_DEFAULT)

        cv.Canny(blurred, edge, 75, 200)

        cv.findContours(edge, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        let maxArea = 0
        let biggestApprox = null

        for (let i = 0; i < contours.size(); ++i) {
            const c = contours.get(i)
            const area = cv.contourArea(c)

            if (area > (src.cols * src.rows * 0.1)) {
                const peri = cv.arcLength(c, true)
                const approx = new cv.Mat()
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

function groupBubblesIntoRows(bubbles: Bubble[], choicesPerQuestion: number): BubbleRow[] {
    bubbles.sort((a, b) => a.centerY - b.centerY)

    const avgHeight = bubbles.reduce((acc, b) => acc + b.height, 0) / bubbles.length
    const rowTolerance = avgHeight * 0.6

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

    for (const row of rows) {
        if (row.bubbles.length >= choicesPerQuestion) {
             let i = 0
            while (i + choicesPerQuestion <= row.bubbles.length) {
                const chunk = row.bubbles.slice(i, i + choicesPerQuestion)
                validRows.push({
                    bubbles: chunk,
                    avgY: row.avgY
                })
                i += choicesPerQuestion
            }
        }
    }

    if (validRows.length === 0) return []

    const sortedByX = [...validRows].sort((a,b) => a.bubbles[0].centerX - b.bubbles[0].centerX)

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

    // IMPORTANT: .roi() creates a view into the parent Mat. The data is NOT continuous.
    // We must clone it to get a continuous block of memory for linear iteration.
    const roiView = gray.roi(roiRect)
    const roi = roiView.clone() // Now continuous
    roiView.delete() // Don't need the view anymore

    const roiThresh = new cv.Mat()
    const mask = new cv.Mat.zeros(roi.rows, roi.cols, cv.CV_8UC1)

    try {
        // Calculate StdDev to detect "flat" areas (paper only)
        // Filled bubbles usually have higher variance (ink vs paper)
        // Empty bubbles (just paper/noise) have low variance
        const mean = new cv.Mat()
        const stddev = new cv.Mat()
        cv.meanStdDev(roi, mean, stddev)
        const std = stddev.data64F[0]
        mean.delete()
        stddev.delete()

        // Threshold for "flatness".
        // A digital empty bubble might be 0 stddev.
        // A camera empty bubble might be ~5-10 stddev due to noise.
        // A filled bubble will be significantly higher.
        if (std < 10) {
            return 0
        }

        // Otsu's Thresholding
        const thresholdVal = cv.threshold(roi, roiThresh, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        // CONTRAST CHECK:
        // Otsu will find a threshold even in noise. We must check if the "ink" is actually dark.
        // We compare the mean of the Foreground (Ink) vs Background (Paper)
        // Or simply: check if the threshold value is sufficiently low?
        // No, threshold value depends on lighting.
        // Better: Check the mean intensity of the "Ink" pixels.

        // Calculate mean of pixels that are classified as Ink (255 in roiThresh)
        let inkSum = 0
        let inkCount = 0
        let paperSum = 0
        let paperCount = 0

        const roiData = roi.data // Original Gray
        const threshData = roiThresh.data // Binarized

        // Also create circular mask
        const center = new cv.Point(Math.floor(roi.cols / 2), Math.floor(roi.rows / 2))
        const r = Math.floor(Math.min(roi.cols, roi.rows) / 2)
        cv.circle(mask, center, r, new cv.Scalar(255), -1)
        const maskData = mask.data

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
            // Homogeneous bubble (all dark or all light).
            // If all dark (inkCount > 0), it's filled.
            // But Otsu usually forces a split. If one is 0, it means it's extremely uniform.
            // Usually means empty or fully filled.
            // If fully filled, inkSum/inkCount should be low (dark).
             return (inkCount > 0) ? 1.0 : 0.0
        }

        const meanInk = inkSum / inkCount
        const meanPaper = paperSum / paperCount

        // Contrast = Light - Dark
        const contrast = meanPaper - meanInk

        // CONTRAST CHECK: Increased to 20 to strictly filter out shadows/noise
        // Empty bubbles with gradients might have contrast ~10-15.
        // Filled bubbles usually have contrast > 40-50.
        if (contrast < 20) {
            return 0
        }

        return inkCount / (inkCount + paperCount)

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

  // Confidence Threshold for Multiple Selections
  // If multiple bubbles are detected, we check if at least one is "strong" (high confidence).
  // If all detected bubbles are "weak" (hovering near minThreshold), we treat the row as noise (Unanswered).
  // e.g., if minThreshold is 0.25, confidence is max(0.40, 0.25+0.15).
  // This prevents shadow noise (e.g., all bubbles ~0.26-0.30) from triggering "Multiple Answer" penalty.
  const confidenceThreshold = Math.max(minThreshold + 0.15, 0.40)

  if (aboveThreshold.length > 1) {
    const sorted = [...aboveThreshold].sort((a, b) => b.fill - a.fill)

    // NOISE FILTER: If the highest fill is still below confidence threshold, treat as Unanswered.
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
        const radius = Math.round(Math.max(bubble.width, bubble.height) / 2 + 2)
        cv.circle(src, center, radius, color, thickness)
    }

    const drawCross = (bubble: Bubble, color: any, thickness: number = 2) => {
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
        .filter(item => item.p >= minThreshold)
        .map(item => item.i)

    if (status === "correct" && correctIdx !== undefined && bubbles[correctIdx]) {
        drawCircle(bubbles[correctIdx], GREEN, 3)
    } else if (status === "incorrect") {
        // Draw Cross (X) for incorrect selections
        if (hasMultiple) {
            selectedIndices.forEach(idx => {
                if (bubbles[idx]) drawCross(bubbles[idx], RED, 3)
            })
        } else if (isValid && selectedIndices.length > 0) {
            const wrongIdx = selectedIndices[0]
            if (bubbles[wrongIdx]) drawCross(bubbles[wrongIdx], RED, 3)
        }

        // Always show correct answer
        if (correctIdx !== undefined && bubbles[correctIdx]) {
            drawCircle(bubbles[correctIdx], GREEN, 3)
        }
    } else if (status === "unanswered") {
        if (correctIdx !== undefined && bubbles[correctIdx]) {
             drawCircle(bubbles[correctIdx], ORANGE, 2)
        }
    }
}
