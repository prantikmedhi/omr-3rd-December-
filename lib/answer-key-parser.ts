/**
 * Parses answer key from CSV text
 * Expected format: question_number,correct_answer_index
 * where correct_answer_index is 1-based (A=1, B=2, etc.)
 */
export function parseAnswerKey(csvText: string): Record<number, number> {
  const answerKey: Record<number, number> = {}
  const lines = csvText.trim().split("\n")

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim()
    if (!line) continue

    const parts = line.split(",").map((p) => p.trim())
    if (parts.length < 2) continue

    // Skip header row
    if (isNaN(Number.parseInt(parts[0])) || isNaN(Number.parseInt(parts[1]))) {
      continue
    }

    const questionNumber = Number.parseInt(parts[0])
    const correctAnswer = Number.parseInt(parts[1])

    if (!isNaN(questionNumber) && !isNaN(correctAnswer)) {
      // Store as 0-based index internally (convert from 1-based input)
      answerKey[questionNumber] = correctAnswer - 1
    }
  }

  return answerKey
}
