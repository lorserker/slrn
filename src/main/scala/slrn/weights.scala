package slrn.weights

import slrn.feature.Feature
import collection.mutable
import scala.util.hashing.MurmurHash3

trait FeatureIndexer {
  def apply(ftr: Feature): Int
}

class VocabularyIndexer extends FeatureIndexer {

  private val vocab = mutable.Map[Feature, Int]()

  def size = vocab.size

  override def apply(ftr: Feature): Int =  {
    if (vocab contains ftr) {
      vocab(ftr)
    } else {
      val i = vocab.size
      vocab(ftr) = i
      assert(vocab.size == i+1)
      i
    }
  }
}

class HashIndexer(val n: Int) extends FeatureIndexer {
  override def apply(ftr: Feature): Int = {
    val hashLong = MurmurHash3.productHash(ftr).toLong + Int.MaxValue + 1
    (hashLong % n).toInt
  }
}

trait Weights {
  def apply(ftr: Feature): Double
  def update(ftr: Feature, w: Double)

  def dot(ftrs: Set[Feature]): Double = ftrs.map(ftr => apply(ftr) * ftr.value).sum
}

object Weights {
  def createInitLike(weights: Weights, initVal: Double = 0.0): Weights = {
      weights match {
        case CompositeWeights(dWgts, nmWgts, _) => CompositeWeights(
          Weights.createInitLike(dWgts, initVal),
          nmWgts.map{case (k, v) => (k, Weights.createInitLike(v, initVal))},
          initVal
        )
        case VocabWeights(ixer, _) => VocabWeights(ixer, initVal)
        case HashWeights(ixer, _) => HashWeights(ixer, initVal)
      }
  }
}

case class CompositeWeights(defaultWeights: Weights, namedWeights: Map[String, Weights], initVal: Double = 0.0) extends  Weights {
  override def apply(ftr: Feature): Double = {
    namedWeights.getOrElse(ftr.name, defaultWeights)(ftr)
  }

  override def update(ftr: Feature, w: Double): Unit = {
    namedWeights.getOrElse(ftr.name, defaultWeights)(ftr) = w
  }
}

case class VocabWeights(ixer: VocabularyIndexer, initVal: Double = 0.0) extends Weights {
  private val storage = mutable.ArrayBuffer[Double]().padTo(ixer.size, initVal)

  def apply(ftr: Feature): Double = {
    val i = ixer(ftr)
    if (i < storage.length) {
      storage(i)
    } else {
      // we have a previously unseen feature
      assert(i == storage.length)
      storage.append(initVal)
      initVal
    }
  }

  def update(ftr: Feature, w: Double): Unit = {
    val i = ixer(ftr)
    if (i < storage.length) {
      storage(i) = w
    } else {
      assert(i == storage.length)
      storage.append(w)
    }
  }
}

case class HashWeights(ixer: HashIndexer, initVal: Double = 0.0) extends Weights {
  private val storage = (new Array[Double](ixer.n)).map(_ + initVal)

  def apply(ftr: Feature): Double = storage(ixer.apply(ftr))

  def update(ftr: Feature, w: Double): Unit = {
    storage(ixer.apply(ftr)) = w
  }

}