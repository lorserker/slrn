package slrn.metrics

trait Metric {
  def add(target: Double, prediction: Double): Unit
  def get(): Double
}

/**
  * NE proposed in 'Practical Lessons from Predicting Clicks on Ads at Facebook'
  * by Xinran He et. al.
  */
class NormalizedEntropy extends Metric {
  var nPos = 0
  var nNeg = 0
  var S = 0.0

  def n = nPos + nNeg

  def avgRate = nPos.toDouble / (nPos + nNeg)

  override def get(): Double = {
    val numerator = S/n
    val p = avgRate
    val denominator = if (nPos == 0 || nNeg == 0) -1.0 else p * log2(p) + (1 - p) * log2(1 - p)
    numerator / denominator
  }

  override def add(target: Double, prediction: Double): Unit = {
    val targetLabel = target.toInt
    nPos += targetLabel
    nNeg += (1 - targetLabel)

    // TODO: this has to be fixed
    if (math.abs(targetLabel - 0.0) < 1e-6) {
      S += (1 - targetLabel) * log2(1 - prediction)
    } else {
      S += targetLabel * log2(prediction)
    }
  }

  private def log2(x: Double): Double = math.log10(x)/math.log10(2.0)
}

class RootMeanSquareError extends Metric {

  private var n = 0
  private var sqErrSum = 0.0

  override def get(): Double = if (n > 0) math.sqrt(sqErrSum / n) else 0.0

  override def add(target: Double, prediction: Double): Unit = {
    n += 1
    sqErrSum += math.pow(target - prediction, 2.0)
  }
}
