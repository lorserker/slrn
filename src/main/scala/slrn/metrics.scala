package slrn.metrics

/**
  * NE proposed in 'Practical Lessons from Predicting Clicks on Ads at Facebook'
  * by Xinran He et. al.
  */
class NormalizedEntropy {
  var nPos = 0
  var nNeg = 0
  var S = 0.0

  def n = nPos + nNeg

  def avgRate = nPos.toDouble / (nPos + nNeg)

  def get(): Double = {
    val numerator = S/n
    val p = avgRate
    val denominator = p * log2(p) + (1 - p) * log2(1 - p)
    numerator / denominator
  }

  def add(target: Int, prediction: Double): Unit = {
    nPos += target
    nNeg += (1 - target)

    if (target == 0) {
      S += (1 - target) * log2(1 - prediction)
    } else {
      S += target * log2(prediction)
    }
  }

  private def log2(x: Double): Double = math.log10(x)/math.log10(2.0)
}