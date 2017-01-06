package slrn.sampling

import slrn.feature._
import org.scalatest.FunSuite

class SamplingTest extends FunSuite {
  test("sampling should work correctly") {
    val biasFtrFun = (ftrs: Set[Feature]) => ftrs.filter(_.name == "bias").toSeq.head

    val nSampled = Array(0, 0)
    val sampler = new Sampler(targetRate = 0.5, biasFtrFun)

    val origRate = 0.01
    var nPos = 0
    val N = 1000000
    for (i <- 1 to N) {
      val label = if (math.random() < origRate) 1 else 0
      if(sampler((label, Set(ContinuousFeature("bias")(1.0))))) {
        if (i > 1000) {
          nSampled(label) += 1
        }
      }
      nPos += label
    }

    val sampledRate = nSampled(1).toDouble / nSampled.sum

    assert(sampledRate > 0.45 && sampledRate < 0.55)

    val correction = sampler.correction(ContinuousFeature("bias")(0.0))
    val correctedPrediction = 1.0 / (1.0 + math.exp(-correction))

    assert(correctedPrediction > 0.008 && correctedPrediction < 0.012)
  }
}