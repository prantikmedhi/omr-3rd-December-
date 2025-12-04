# OMR Sheet Scanner

A powerful web-based OMR (Optical Mark Recognition) sheet analysis tool that automatically detects and evaluates answer bubbles on scanned exam sheets.

## Features

- **Automatic Bubble Detection**: Uses advanced image processing (Sobel edge detection) to identify answer bubbles
- **4-Choice Question Support**: Configure for 2-10 choices per question (default: 4)
- **Answer Key Comparison**: Upload a CSV file with correct answers for instant evaluation
- **Real-time Results**: Get detailed scoring with visual feedback on the processed sheet
- **Darkness Detection**: Recognizes darker circles as marked answers, lighter ones as unmarked
- **Visual Indicators**: Shows correct answers (green), incorrect (red), unanswered (orange), and multiple selections (yellow)

## How It Works

### Image Processing Pipeline

1. **Grayscale Conversion**: Converts image to grayscale for uniform processing
2. **Otsu Thresholding**: Automatically calculates optimal threshold for binary classification
3. **Edge Detection**: Uses Sobel filter to find circular shapes
4. **Bubble Grouping**: Organizes detected circles into question rows
5. **Question Number Filtering**: Automatically skips the first row if it contains fewer bubbles than expected (question numbers)
6. **Fill Calculation**: Measures darkness percentage inside each bubble to determine if marked
7. **Answer Matching**: Compares detected marks with provided answer key

### Key Improvements

- **Question Number Exclusion**: The first row with fewer bubbles than expected is automatically excluded
- **Adaptive Bubble Detection**: Filters out non-answer bubbles using size and shape analysis
- **Robust Threshold**: Otsu's method ensures accurate detection across different image qualities

## Usage

### 1. Prepare OMR Sheet Image
- Scan or photograph a completed OMR sheet
- Ensure clear visibility of bubbles
- Format: PNG, JPG, or other common image formats

### 2. Create Answer Key CSV
Format: `question_number,correct_answer`
\`\`\`csv
1,1
2,3
3,2
4,4
\`\`\`
- Question numbers start from 1
- Answer indices: 1=A, 2=B, 3=C, 4=D, etc.

### 3. Configure Settings
- **Choices per Question**: Set to match your OMR sheet (2-10)
- **Minimum Fill Threshold**: Lower value (10-15%) for lightly marked bubbles, higher (25-30%) for stricter detection

### 4. Process and Analyze
- Upload both files
- Click "Process OMR Sheet"
- View results with visual feedback

## Result Information

### Score Summary
- **Score**: Number of correct answers
- **Percentage**: Accuracy percentage

### Statistics
- **Correct**: Correctly marked answers (✓ green)
- **Wrong**: Incorrect answers (✗ red)
- **Blank**: Unanswered questions (- orange)
- **Multiple**: Questions with multiple marks (⚠ yellow)

### Question Details
- Shows each question's result status
- Displays selected answer vs correct answer
- Visual indicators for quick reference

## Technical Details

### Bubble Detection Algorithm
- Minimum bubble size: ~1.5% of image width
- Maximum bubble size: ~6% of image width
- Accepts bubbles with 40-130% fill of circular area (accounts for imperfect circles and pen marks)

### Fill Percentage Calculation
- Samples pixels inside an ellipse (0.35× bubble dimensions)
- Compares pixel darkness to Otsu threshold
- Percentage = dark pixels / total pixels

### Row Grouping
- Groups bubbles by vertical position
- Filters outlier rows (different sizes)
- Excludes question number rows automatically

## Troubleshooting

### No bubbles detected
- Ensure image is clear and well-lit
- Check that bubbles are visible circles
- Try adjusting minimum fill threshold

### Question numbers being detected
- ✓ Fixed: First row with fewer bubbles is now automatically skipped
- Check "Choices per Question" setting

### Wrong answers detected
- Adjust "Minimum Fill Threshold"
- Lower for lightly marked bubbles
- Higher for stricter marking detection

### Multiple answers selected
- This is correctly detected when >1 bubble meets threshold
- Review the visual feedback for guidance

## Deployment

To deploy to Vercel:

1. Ensure all dependencies are installed:
   \`\`\`bash
   npm install
   \`\`\`

2. (Optional) For analytics, install:
   \`\`\`bash
   npm install @vercel/analytics
   \`\`\`

3. Deploy:
   \`\`\`bash
   vercel deploy
   \`\`\`

## Notes

- The tool automatically excludes question number rows based on bubble count
- Supports different lighting conditions and image qualities
- Best results with standard black pen marks on white paper
- Works with various OMR sheet designs and layouts
