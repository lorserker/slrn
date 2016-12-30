package slrn.transform

import slrn.feature._

import collection.mutable
import scala.util.{Success, Try}


trait Transform {
  def apply(ftrs: Set[Feature]): Set[Feature]
}

object UnitVector extends Transform {
  def apply(ftrs: Set[Feature]): Set[Feature] = {
    val s = math.sqrt(ftrs.map(ftr => ftr.value * ftr.value).sum)
    if (s > 0) {
      ftrs.map(ftr => Feature.withValue(ftr, ftr.value / s))
    } else {
      ftrs
    }
  }
}

class Scaler extends Transform {

  private val stats: mutable.Map[Feature, OnlineMeanStd] = mutable.Map.empty

  def apply(ftrs: Set[Feature]): Set[Feature] = {
    val scaledFtrs = ftrs.map {
      case ftr@NumericFeature(name) =>
        val result = if (stats contains ftr) {
          val mu = stats(ftr).mean
          Try(stats(ftr).variance) match {
            case Success(variance) => {
              val std = math.sqrt(variance)
              if (std > 1e-3) Feature.withValue(ftr, (ftr.value - mu) / std) else Feature.withValue(ftr, mu)
            }
            case _ => ftr
          }
        } else {
          stats(ftr) = new OnlineMeanStd
          ftr
        }
        stats(ftr).add(ftr.value)
        result
      case ftr => ftr
    }
    scaledFtrs
  }
}

// implements this algorithm for computing online variance
// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
class OnlineMeanStd {
  private var count = 0
  private var s = 0.0
  private var m2 = 0.0

  def n = count

  def add(x: Double): Unit = {
    val oldMean = mean
    count += 1
    s += x
    m2 = m2 + (x - oldMean)*(x - mean)
  }

  def variance: Double = {
    if (n < 2) {
      throw new Exception("undefined variance")
    } else {
      m2 / n
    }
  }

  def mean: Double = if (n < 1) 0 else s / n
}

object ApplyAll {
  def apply(transforms: List[Transform])(ftrs: Set[Feature]): Set[Feature] = {
    transforms match {
      case Nil => ftrs
      case first :: rest => apply(rest)(first(ftrs))
    }
  }
}
