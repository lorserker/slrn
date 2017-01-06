package slrn.sampling

import slrn.feature.Feature
import slrn.weights.Weights

import collection.mutable


class Sampler(targetRate: Double, biasFeatureSelect: Set[Feature] => Feature, decay: Double = 1e-5) {
  require(targetRate > 0)

  val groupStats = mutable.Map[Feature, (Double, Double)]()

  def apply(labeledFtrs: (Double, Set[Feature])): Boolean = {
    val biasFtr = biasFeatureSelect(labeledFtrs._2)
    val (nPositive, nNegative) = groupStats.getOrElse(biasFtr, (1.0, 1.0))
    val label = labeledFtrs._1
    groupStats(biasFtr) = (
      nPositive*(1-decay) + label,
      nNegative*(1-decay) + 1 - label
    )

    if (nPositive < nNegative) {
      if (label > 0) {
        true
      } else {
        val rate = ((1-targetRate)*nPositive)/(targetRate*nNegative)
        math.random() < rate
      }
    } else {
      if(label < 1) {
        true
      } else {
        val rate = (targetRate * nNegative) / (nPositive * (1-targetRate))
        math.random() < rate
      }
    }
  }

  def correction(ftr: Feature): Double = {
      if (groupStats contains ftr) {
        val (nPositive, nNegative) = groupStats.getOrElse(ftr, (1.0, 1.0))
        val originalRate = nPositive / (nPositive + nNegative)
         - math.log((targetRate*(1-originalRate))/(originalRate*(1-targetRate)))
      } else {
        0.0
      }
  }
}

class PriorCorrectedWeights(weights: Weights, sampler: Sampler) extends Weights {
  override def apply(ftr: Feature): Double = weights.apply(ftr)

  override def update(ftr: Feature, w: Double): Unit = weights.update(ftr, w)

  override def dot(ftrs: Set[Feature]): Double = {
    ftrs.map(ftr => (apply(ftr) + sampler.correction(ftr)) * ftr.value).sum
  }
}

