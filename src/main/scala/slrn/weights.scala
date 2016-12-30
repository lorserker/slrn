package slrn.weights

import slrn.feature.Feature
import collection.mutable

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
      i
    }
  }
}

class HashIndexer(val n: Int) extends FeatureIndexer {
  override def apply(ftr: Feature): Int = {
    val hashLong = ftr.hashCode.toLong + Int.MaxValue + 1
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
        case BlockWeights(wBlocks, dWeights, ftr2block) => {
          val newWBlocks = wBlocks.map(wgts => Weights.createInitLike(wgts, initVal))
          BlockWeights(newWBlocks, Weights.createInitLike(dWeights, initVal), ftr2block)
        }
        case VocabWeights(ixer, _) => VocabWeights(ixer, initVal)
        case HashWeights(ixer, _) => HashWeights(ixer, initVal)
      }
  }
}

case class BlockWeights(weightBlocks: Array[Weights], defaultWeights: Weights, ftr2block: Feature => Int) extends Weights {
  // require that each weights object reference (in weightBlocks and defaultWeights) be unique
  val weightsNotUnique = {
    (weightBlocks.length == 1 && (defaultWeights eq weightBlocks(0))) ||
    (for {
      i <- 0 until weightBlocks.length - 1
      j <- i + 1 until weightBlocks.length
    } yield {
      (defaultWeights eq weightBlocks(i)) || (defaultWeights eq weightBlocks(j)) || (weightBlocks(i) eq weightBlocks(j))
    }).reduce(_ || _)
  }
  require(!weightsNotUnique, "all Weights objects in a BlockWeights should be unique")

  override def apply(ftr: Feature): Double = {
    val blockIndex = ftr2block(ftr)
    if (blockIndex < 0 || blockIndex >= weightBlocks.length) {
      defaultWeights(ftr)
    } else {
      weightBlocks(blockIndex)(ftr)
    }
  }

  override def update(ftr: Feature, w: Double): Unit = {
    val blockIndex = ftr2block(ftr)
    if (blockIndex < 0 || blockIndex >= weightBlocks.length) {
      defaultWeights(ftr) = w
    } else {
      weightBlocks(blockIndex)(ftr) = w
    }
  }
}

case class VocabWeights(indexer: VocabularyIndexer, initVal: Double = 0.0) extends Weights {
  private val storage = mutable.ArrayBuffer[Double]().padTo(indexer.size, initVal)

  def apply(ftr: Feature): Double = {
    val i = indexer(ftr)
    if (i < storage.length) {
      storage(i)
    } else {
      extendStorage(i + 1)
      storage(i)
    }
  }

  def update(ftr: Feature, w: Double): Unit = {
    val i = indexer(ftr)
    if (i < storage.length) {
      storage(i) = w
    } else {
      extendStorage(i + 1)
      storage(i) = w
    }
  }

  private def extendStorage(toLength: Int): Unit = {
    val values = for (i <- storage.length until toLength) yield initVal
    storage.append(values:_*)
  }
}

case class HashWeights(indexer: HashIndexer, initVal: Double = 0.0) extends Weights {
  private val storage = new Array[Double](indexer.n).map(_ + initVal)

  def apply(ftr: Feature): Double = storage(indexer.apply(ftr))

  def update(ftr: Feature, w: Double): Unit = {
    storage(indexer.apply(ftr)) = w
  }
}