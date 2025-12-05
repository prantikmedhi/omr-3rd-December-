/**
 * OpenCV.js Type Definitions
 * Provides type safety for OpenCV.js operations
 */

export interface OpenCVMat {
  cols: number
  rows: number
  data: Uint8Array | Float32Array | Int32Array
  data8S?: Int8Array
  data8U?: Uint8Array
  data16S?: Int16Array
  data16U?: Uint16Array
  data32S?: Int32Array
  data32F?: Float32Array
  data64F?: Float64Array
  delete(): void
  clone(): OpenCVMat
  roi(rect: OpenCVRect): OpenCVMat
  copyTo(dst: OpenCVMat, mask?: OpenCVMat): void
}

export interface OpenCVRect {
  x: number
  y: number
  width: number
  height: number
}

export interface OpenCVPoint {
  x: number
  y: number
}

export interface OpenCVSize {
  width: number
  height: number
}

export interface OpenCVScalar {
  data: number[]
}

export interface OpenCVMatVector {
  size(): number
  get(index: number): OpenCVMat
  push_back(mat: OpenCVMat): void
  delete(): void
}

export interface OpenCVContour {
  data32S?: Int32Array
  rows: number
}

export interface OpenCVInstance {
  // Mat creation
  Mat: {
    new (rows?: number, cols?: number, type?: number): OpenCVMat
    ones(rows: number, cols: number, type: number): OpenCVMat
    zeros(rows: number, cols: number, type: number): OpenCVMat
  }
  
  // Constants
  COLOR_RGBA2GRAY: number
  COLOR_BGR2GRAY: number
  COLOR_GRAY2RGBA: number
  CV_8UC1: number
  CV_8UC3: number
  CV_8UC4: number
  CV_32FC1: number
  CV_32FC2: number
  CV_32S: number
  
  // Threshold types
  THRESH_BINARY: number
  THRESH_BINARY_INV: number
  THRESH_OTSU: number
  ADAPTIVE_THRESH_MEAN_C: number
  ADAPTIVE_THRESH_GAUSSIAN_C: number
  
  // Morphology
  MORPH_CLOSE: number
  MORPH_OPEN: number
  MORPH_DILATE: number
  MORPH_ERODE: number
  MORPH_GRADIENT: number
  MORPH_TOPHAT: number
  MORPH_BLACKHAT: number
  
  // Contour retrieval
  RETR_EXTERNAL: number
  RETR_LIST: number
  RETR_TREE: number
  CHAIN_APPROX_SIMPLE: number
  CHAIN_APPROX_NONE: number
  
  // Border types
  BORDER_DEFAULT: number
  BORDER_CONSTANT: number
  BORDER_REPLICATE: number
  
  // Interpolation flags (may not be available in all OpenCV.js builds)
  INTER_LINEAR?: number
  INTER_NEAREST?: number
  
  // Image operations
  imread(canvas: HTMLCanvasElement | HTMLImageElement): OpenCVMat
  imshow(canvas: HTMLCanvasElement, mat: OpenCVMat): void
  
  // Color conversion
  cvtColor(src: OpenCVMat, dst: OpenCVMat, code: number, dstCn?: number): void
  
  // Blur
  GaussianBlur(
    src: OpenCVMat,
    dst: OpenCVMat,
    ksize: OpenCVSize,
    sigmaX: number,
    sigmaY: number,
    borderType?: number
  ): void
  blur(
    src: OpenCVMat,
    dst: OpenCVMat,
    ksize: OpenCVSize,
    anchor?: OpenCVPoint,
    borderType?: number
  ): void
  medianBlur(src: OpenCVMat, dst: OpenCVMat, ksize: number): void
  
  // Thresholding
  threshold(
    src: OpenCVMat,
    dst: OpenCVMat,
    thresh: number,
    maxval: number,
    type: number
  ): number
  adaptiveThreshold(
    src: OpenCVMat,
    dst: OpenCVMat,
    maxValue: number,
    adaptiveMethod: number,
    thresholdType: number,
    blockSize: number,
    C: number
  ): void
  
  // Morphology
  morphologyEx(
    src: OpenCVMat,
    dst: OpenCVMat,
    op: number,
    kernel: OpenCVMat,
    anchor?: OpenCVPoint,
    iterations?: number,
    borderType?: number,
    borderValue?: OpenCVScalar
  ): void
  
  // Edge detection
  Canny(
    image: OpenCVMat,
    edges: OpenCVMat,
    threshold1: number,
    threshold2: number,
    apertureSize?: number,
    L2gradient?: boolean
  ): void
  Sobel(
    src: OpenCVMat,
    dst: OpenCVMat,
    ddepth: number,
    dx: number,
    dy: number,
    ksize?: number,
    scale?: number,
    delta?: number,
    borderType?: number
  ): void
  
  // Contours
  findContours(
    image: OpenCVMat,
    contours: OpenCVMatVector,
    hierarchy: OpenCVMat,
    mode: number,
    method: number,
    offset?: OpenCVPoint
  ): void
  contourArea(contour: OpenCVMat, oriented?: boolean): number
  arcLength(curve: OpenCVMat, closed: boolean): number
  approxPolyDP(
    curve: OpenCVMat,
    approxCurve: OpenCVMat,
    epsilon: number,
    closed: boolean
  ): void
  isContourConvex(contour: OpenCVMat): boolean
  convexHull(
    contour: OpenCVMat,
    hull: OpenCVMat,
    clockwise?: boolean,
    returnPoints?: boolean
  ): void
  boundingRect(array: OpenCVMat): OpenCVRect
  
  // Geometric transforms
  getPerspectiveTransform(src: OpenCVMat, dst: OpenCVMat): OpenCVMat
  warpPerspective(
    src: OpenCVMat,
    dst: OpenCVMat,
    M: OpenCVMat,
    dsize: OpenCVSize,
    flags?: number,
    borderMode?: number,
    borderValue?: OpenCVScalar
  ): void
  rotate(
    src: OpenCVMat,
    dst: OpenCVMat,
    rotateCode: number
  ): void
  ROTATE_90_CLOCKWISE: number
  ROTATE_180: number
  ROTATE_90_COUNTERCLOCKWISE: number
  getRotationMatrix2D(
    center: OpenCVPoint,
    angle: number,
    scale: number
  ): OpenCVMat
  warpAffine(
    src: OpenCVMat,
    dst: OpenCVMat,
    M: OpenCVMat,
    dsize: OpenCVSize,
    flags?: number,
    borderMode?: number,
    borderValue?: OpenCVScalar
  ): void
  
  // Drawing
  circle(
    img: OpenCVMat,
    center: OpenCVPoint,
    radius: number,
    color: OpenCVScalar,
    thickness?: number,
    lineType?: number,
    shift?: number
  ): void
  line(
    img: OpenCVMat,
    pt1: OpenCVPoint,
    pt2: OpenCVPoint,
    color: OpenCVScalar,
    thickness?: number,
    lineType?: number,
    shift?: number
  ): void
  rectangle(
    img: OpenCVMat,
    pt1: OpenCVPoint,
    pt2: OpenCVPoint,
    color: OpenCVScalar,
    thickness?: number,
    lineType?: number,
    shift?: number
  ): void
  
  // Statistics
  meanStdDev(
    src: OpenCVMat,
    mean: OpenCVMat,
    stddev: OpenCVMat,
    mask?: OpenCVMat
  ): void
  minMaxLoc(
    src: OpenCVMat,
    minVal?: { value: number },
    maxVal?: { value: number },
    minLoc?: OpenCVPoint,
    maxLoc?: OpenCVPoint,
    mask?: OpenCVMat
  ): void
  
  // Utility
  matFromArray(
    rows: number,
    cols: number,
    type: number,
    array: number[]
  ): OpenCVMat
  Point: {
    new (x: number, y: number): OpenCVPoint
  }
  Size: {
    new (width: number, height: number): OpenCVSize
  }
  Scalar: {
    new (v0: number, v1?: number, v2?: number, v3?: number): OpenCVScalar
  }
  MatVector: {
    new (): OpenCVMatVector
  }
  Rect: {
    new (x: number, y: number, width: number, height: number): OpenCVRect
  }
  
  // Build info
  getBuildInformation?(): string
  onRuntimeInitialized?: () => void
}

declare global {
  interface Window {
    cv: OpenCVInstance
  }
  const cv: OpenCVInstance
}


