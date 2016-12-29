package slrn

package model {

  import java.io.PrintWriter

  import slrn.feature._

  import collection.mutable

  trait Model {

    def dot(ftrs: Set[Feature]): Double = ftrs.map(ftr => ftr.value * weight(ftr)).sum

    def weight(ftr: Feature): Double

    def setWeight(ftr: Feature, w: Double)

    def predict(ftrs: Set[Feature]): Double = 1.0 / (1.0 + math.exp(-dot(ftrs)))

    def save(out: java.io.OutputStream)

  }

  class DictionaryWeights extends Model {

    private val weightsMap = mutable.Map[String, mutable.Map[Feature, Double]]()

    override def weight(ftr: Feature): Double = {
      weightsMap.getOrElse(ftr.name, Map[Feature, Double]()).getOrElse(ftr, 0.0)
    }

    override def setWeight(ftr: Feature, w: Double): Unit = {
      if (weightsMap contains ftr.name) {
        weightsMap(ftr.name)(ftr) = w
      } else {
        weightsMap(ftr.name) = mutable.Map(ftr -> w)
      }
    }

    override def save(out: java.io.OutputStream): Unit = {
      val writer = new PrintWriter(out)
      try {
        for ((ftrName, ftrWeights) <- weightsMap) {
          for ((ftr, w) <- ftrWeights) {
              ftr match {
                case NumericFeature(name) => writer.println(s"${name}\t$w")
                case CategoricFeature(name, nominal) => writer.println(s"$name\t$nominal\t$w")
              }
          }
        }
      } catch {
        case ex: Exception => throw ex
      } finally {
        writer.close()
      }
    }
  }

  trait OnlineLearner {
    def model: Model

    def learn(label: Int, ftrs: Set[Feature])

    def update(ftr: Feature, gradient: Double)

    def gradient(ftr: Feature, trueValue: Int, predictedValue: Double): Double = (predictedValue - trueValue)*ftr.value
  }

  class ConstantStepSGD(learningRate: Double, val model: Model) extends OnlineLearner {
    require(learningRate > 0.0)

    override def learn(label: Int, ftrs: Set[Feature]): Unit = {
      val p = model.predict(ftrs)
      require(p > 0.0 && p < 1.0)
      for (ftr <- ftrs) {
        update(ftr, gradient(ftr, label, p))
      }
    }

    override def update(ftr: Feature, grad: Double) {
      model.setWeight(ftr, model.weight(ftr) - learningRate*grad)
    }
  }

  class MiniBatch(batchSize: Int, val learner: OnlineLearner) extends OnlineLearner {
    require(batchSize > 0)

    def model = learner.model

    private var n = 0
    private val examples = new Array[(Int, Set[Feature])](batchSize)

    override def learn(label: Int, ftrs: Set[Feature]): Unit = {
      if (n < batchSize) {
        // accumulate examples
        examples(n) = (label, ftrs)
        n += 1
      } else {
        n = 0

        val gradientSums = mutable.Map[Feature, Double]()
        val ftrCounts = mutable.Map[Feature, Int]()
        for ((label, ftrs) <- examples) {
          val p = model.predict(ftrs)
          for (ftr <- ftrs) {
            val g = gradient(ftr, label, p)
            gradientSums(ftr) = gradientSums.getOrElseUpdate(ftr, 0.0) + g
            ftrCounts(ftr) = ftrCounts.getOrElse(ftr, 0) + 1
          }
        }

        for ((ftr, gSum) <- gradientSums) {
          update(ftr, gSum / ftrCounts(ftr))
        }
      }
    }

    override def update(ftr: Feature, grad: Double) {
      learner.update(ftr, grad)
    }

  }

  trait LocalGradientStats {
    def update(ftrGrad: Map[Feature, Double])
    def avgGradient(ftr: Feature): Double
    def avgSquareGradient(ftr: Feature): Double
    def avgDiagHess(ftr: Feature): Double
    def learningRate(ftr: Feature): Double
  }

  class DictLocalGradientStats extends LocalGradientStats {
    val gradientEMA = mutable.Map[Feature, Double]()
    val sqrGradientEMA = mutable.Map[Feature, Double]()
    val memoryEMA = mutable.Map[Feature, Double]()
    val hessEMA = mutable.Map[Feature, Double]()
    val prevGradient = mutable.Map[Feature, Double]()

    val epsilon = 1e-3

    override def avgGradient(ftr: Feature): Double = gradientEMA.getOrElse(ftr, 0.1)
    override def avgSquareGradient(ftr: Feature): Double = sqrGradientEMA.getOrElse(ftr, 0.2)
    override def avgDiagHess(ftr: Feature): Double = hessEMA.getOrElse(ftr, 0.05)

    def memory(ftr: Feature): Double = memoryEMA.getOrElse(ftr, 3.0) max 3.0

    override def learningRate(ftr: Feature): Double = 0.001 max (0.3 min (avgGradient(ftr) * avgGradient(ftr) / (avgDiagHess(ftr) * avgSquareGradient(ftr))))

    override def update(ftrGrad: Map[Feature, Double]): Unit = {

      def applyUpdates(newValues: Map[Feature, Double], stats: mutable.Map[Feature, Double]): Unit = {
        for ((k, v) <- newValues) {
          stats(k) = v
        }
      }

      val newGradients = for ((ftr, g) <- ftrGrad) yield {
        val oldG = avgGradient(ftr)
        val tau = memory(ftr)
        require(tau > 0)
        val newG = (1.0 - 1.0/tau) * oldG + g / tau
        (ftr, newG)
      }

      val newSqrGradients = for ((ftr, g) <- ftrGrad) yield {
        val oldSqrG = avgSquareGradient(ftr)
        val tau = memory(ftr)
        require(tau > 0)
        val newSqrG = (1.0 - 1.0/tau) * oldSqrG + g * g / tau
        (ftr, newSqrG)
      }

      val newHess = for ((ftr, g) <- ftrGrad) yield {
        val oldHess = avgDiagHess(ftr)
        val tau = memory(ftr)
        require(tau > 0)
        val newHess = (1.0 - 1.0/tau) * oldHess + math.abs(g - prevGradient.getOrElse(ftr, 0.0)) / tau
        (ftr, newHess max epsilon)
      }

      val newMemory = for((ftr, _) <- ftrGrad) yield {
        val tau = memory(ftr)
        require(tau > 0)
        val avgG = avgGradient(ftr)
        val avgSqG = avgSquareGradient(ftr)
        val sqOfAvgG = avgG * avgG
        require(sqOfAvgG <= avgSqG)
        val newTau = (1 - sqOfAvgG/avgSqG) * tau + 1
        (ftr, newTau)
      }

      applyUpdates(newGradients, gradientEMA)
      applyUpdates(newSqrGradients, sqrGradientEMA)
      applyUpdates(newHess, hessEMA)
      applyUpdates(newMemory, memoryEMA)

      applyUpdates(ftrGrad, prevGradient)
    }
  }

  class LocalVarSGD(val model: Model) extends OnlineLearner {

    val gStats = new DictLocalGradientStats()

    def learn(label: Int, ftrs: Set[Feature]): Unit = {
      val p = model.predict(ftrs)

      val ftrGrad: Map[Feature, Double] = (for (ftr <- ftrs) yield (ftr, gradient(ftr, label, p))).toMap

      for ((ftr, g) <- ftrGrad) {
        update(ftr, g)
      }

      gStats.update(ftrGrad)
    }

    def update(ftr: Feature, grad: Double): Unit = {
      model.setWeight(ftr, model.weight(ftr) - gStats.learningRate(ftr)*grad)
    }
  }

  class AdaDelta(val model: Model, gamma: Double = 0.9, epsilon: Double = 1e-6) extends OnlineLearner {

    val sqrGradEMA = mutable.Map[Feature, Double]()
    val sqrDeltaEMA = mutable.Map[Feature, Double]()

    def learn(label: Int, ftrs: Set[Feature]): Unit = {
      val p = model.predict(ftrs)

      val ftrGrad: Map[Feature, Double] = (for (ftr <- ftrs) yield (ftr, gradient(ftr, label, p))).toMap

      for ((ftr, g) <- ftrGrad) {

        sqrGradEMA(ftr) = gamma * math.sqrt(sqrGradEMA.getOrElse(ftr, 0.1)) + (1 - gamma) * g * g

        val rmsGrad = math.sqrt(sqrGradEMA(ftr)) + epsilon
        val rmsDelta = math.sqrt(sqrDeltaEMA.getOrElse(ftr, 0.05)) + epsilon
        val delta = -g*rmsDelta/rmsGrad

        update(ftr, delta)

        sqrDeltaEMA(ftr) = gamma * sqrDeltaEMA.getOrElse(ftr, 0.05) + (1 - gamma) * delta * delta
      }
    }

    def update(ftr: Feature, delta: Double): Unit = {
      model.setWeight(ftr, model.weight(ftr) + delta)
    }

  }

}