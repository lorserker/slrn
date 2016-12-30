package slrn.model

import slrn.feature._
import collection.mutable
import slrn.weights._

trait Prediction {
  def predict(ftrs: Set[Feature]): Double
}

trait LogisticPrediction extends Weights with Prediction {
  def predict(ftrs: Set[Feature]): Double = 1.0 / (1.0 + math.exp(-dot(ftrs)))
}

trait Gradient {
  def gradient(ftr: Feature, trueValue: Double, predictedValue: Double): Double = (predictedValue - trueValue) * ftr.value
}

trait Learner {
  def model: Weights with Prediction

  def learn(target: Double, ftrs: Set[Feature])

  def updateModel(ftrGradients: Seq[(Feature, Double)])
}

class ConstantStepSGD(learningRate: Double, val model: Weights with Prediction) extends Learner with Gradient {
  require(learningRate > 0.0)

  override def learn(target: Double, ftrs: Set[Feature]): Unit = {
    val p = model.predict(ftrs)
    updateModel(for (ftr <- ftrs.toSeq) yield (ftr, gradient(ftr, target, p)))
  }

  override def updateModel(ftrGradients: Seq[(Feature, Double)]): Unit = {
    for ((ftr, g) <- ftrGradients) {
      model(ftr) = model(ftr) - learningRate * g
    }
  }
}

class MiniBatch(batchSize: Int, val learner: Learner) extends Learner with Gradient {
  require(batchSize > 0)

  def model = learner.model

  private var n = 0
  private val examples = new Array[(Double, Set[Feature])](batchSize)

  override def learn(target: Double, ftrs: Set[Feature]): Unit = {
    if (n < batchSize) {
      // accumulate examples
      examples(n) = (target, ftrs)
      n += 1
    } else {
      n = 0

      val gradientSums = mutable.Map[Feature, Double]()
      val ftrCounts = mutable.Map[Feature, Int]()
      for ((target, ftrs) <- examples) {
        val p = model.predict(ftrs)
        for (ftr <- ftrs) {
          val g = gradient(ftr, target, p)
          gradientSums(ftr) = gradientSums.getOrElse(ftr, 0.0) + g
          ftrCounts(ftr) = ftrCounts.getOrElse(ftr, 0) + 1
        }
      }

      updateModel(gradientSums.toSeq.map { case (ftr, gSum) => (ftr, gSum / ftrCounts(ftr)) })
    }
  }

  override def updateModel(ftrGradients: Seq[(Feature, Double)]) {
    learner.updateModel(ftrGradients)
  }
}

class LocalVarSGD(val model: Weights with Prediction) extends Learner with Gradient {

  private val gStats = new DictLocalGradientStats(model)

  def learn(target: Double, ftrs: Set[Feature]): Unit = {
    val p = model.predict(ftrs)

    val ftrGrad = ftrs.toSeq.map(ftr => (ftr, gradient(ftr, target, p)))

    updateModel(ftrGrad)

    gStats.update(ftrGrad)
  }

  override def updateModel(ftrGradients: Seq[(Feature, Double)]): Unit = {
    for ((ftr, g) <- ftrGradients) {
      model(ftr) = model(ftr) - gStats.learningRate(ftr) * g
    }
  }
}

private class DictLocalGradientStats(model: Weights) {

  val gradientEMA = Weights.createInitLike(model, 0.1)
  val sqrGradientEMA = Weights.createInitLike(model, 0.2)
  val memoryEMA = Weights.createInitLike(model, 3.0)
  val hessEMA = Weights.createInitLike(model, 0.05)
  val prevGradient = Weights.createInitLike(model, 0.0)

  val epsilon = 1e-3

  def avgGradient(ftr: Feature): Double = gradientEMA(ftr)

  def avgSquareGradient(ftr: Feature): Double = sqrGradientEMA(ftr)

  def avgDiagHess(ftr: Feature): Double = hessEMA(ftr)

  def memory(ftr: Feature): Double = memoryEMA(ftr) max 3.0

  def learningRate(ftr: Feature): Double = 0.001 max (0.3 min (avgGradient(ftr) * avgGradient(ftr) / (avgDiagHess(ftr) * avgSquareGradient(ftr))))

  def update(ftrGrad: Seq[(Feature, Double)]): Unit = {

    def applyUpdates(newValues: Seq[(Feature, Double)], stats: Weights): Unit = {
      for ((k, v) <- newValues) {
        stats(k) = v
      }
    }

    val newGradients = for ((ftr, g) <- ftrGrad) yield {
      val oldG = avgGradient(ftr)
      val tau = memory(ftr)
      require(tau > 0)
      val newG = (1.0 - 1.0 / tau) * oldG + g / tau
      (ftr, newG)
    }

    val newSqrGradients = for ((ftr, g) <- ftrGrad) yield {
      val oldSqrG = avgSquareGradient(ftr)
      val tau = memory(ftr)
      require(tau > 0)
      val newSqrG = (1.0 - 1.0 / tau) * oldSqrG + g * g / tau
      (ftr, newSqrG)
    }

    val newHess = for ((ftr, g) <- ftrGrad) yield {
      val oldHess = avgDiagHess(ftr)
      val tau = memory(ftr)
      require(tau > 0)
      val newHess = (1.0 - 1.0 / tau) * oldHess + math.abs(g - prevGradient(ftr)) / tau
      (ftr, newHess max epsilon)
    }

    val newMemory = for ((ftr, _) <- ftrGrad) yield {
      val tau = memory(ftr)
      require(tau > 0)
      val avgG = avgGradient(ftr)
      val avgSqG = avgSquareGradient(ftr)
      val sqOfAvgG = avgG * avgG
      require(sqOfAvgG <= avgSqG)
      val newTau = (1 - sqOfAvgG / avgSqG) * tau + 1
      (ftr, newTau)
    }

    applyUpdates(newGradients, gradientEMA)
    applyUpdates(newSqrGradients, sqrGradientEMA)
    applyUpdates(newHess, hessEMA)
    applyUpdates(newMemory, memoryEMA)

    applyUpdates(ftrGrad, prevGradient)
  }
}
