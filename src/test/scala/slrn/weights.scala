package slrn.weights

import slrn.feature._
import org.scalatest.FunSuite

class FeatureIndexerTest extends FunSuite {

  test("vocab indexer should work properly") {
    val vocabIndexer: FeatureIndexer = new VocabularyIndexer
    val n = 1000
    for (number <- 0 until n) {
      val i = vocabIndexer.apply(CategoricFeature("ftr1", number.toString)(1.0))
      assert(i == number)
      assert(i == vocabIndexer.apply(CategoricFeature("ftr1", number.toString)(1.0)))
      assert(i == vocabIndexer.apply(CategoricFeature("ftr1", number.toString)(5.0)))
    }
    for (number <- 0 until n) {
      val i = vocabIndexer.apply(CategoricFeature("ftr2", number.toString)(1.0))
      assert(i == number + n)
    }
    assert(vocabIndexer.apply(NumericFeature("ftr1")(1.0)) == 2*n)
    assert(vocabIndexer.apply(NumericFeature("ftr1")(2.0)) == 2*n)
    assert(vocabIndexer.apply(NumericFeature("ftr1")(3.0)) == 2*n)

    assert(vocabIndexer.apply(NumericFeature("ftr2")(1.0)) == 2*n + 1)
  }

  test("hashing indexer should work properly") {
    val n = 10
    val hashIndexer: FeatureIndexer = new HashIndexer(n)

    val indexes = (0 until 100 * n).map(n => CategoricFeature("ftr1", n.toString)(1.0)).map(ftr => hashIndexer.apply(ftr))

    assert(indexes.min == 0)
    assert(indexes.max == n - 1)
  }
}

class WeightsTest extends FunSuite {

  test("weights should work properly") {
    val w: Weights = BlockWeights(
      weightBlocks = Array(
        VocabWeights(new VocabularyIndexer),
        HashWeights(new HashIndexer(10))
      ),
      defaultWeights = VocabWeights(new VocabularyIndexer),
      ftr2block = (ftr: Feature) => {
        Map("color" -> 0, "size" -> 1).getOrElse(ftr.name, -1)
      }
    )

    assert(w(NumericFeature("bla")(1.0)) == 0.0)
    w(NumericFeature("bla")(1.0)) = 1.0
    assert(w(NumericFeature("bla")(1.0)) == 1.0)

    for ((color, i) <- Array("red", "green", "blue").zipWithIndex) {
      val ftr = CategoricFeature("color", color)(1.0)
      assert(w(ftr) == 0.0)
      w(ftr) = i.toDouble
      assert(w(ftr) == i.toDouble)
    }

    for (s <- 1 to 1000) {
      val ftr = CategoricFeature("size", s.toString)(1.0)
      assert(w(ftr) < s.toDouble)
      w(ftr) = s.toDouble
      assert(w(ftr) == s.toDouble)
    }
  }

  test("initial values should work for weights") {
    val initVal = 0.1
    val vocabWeights = VocabWeights(new VocabularyIndexer, initVal)
    val newFtr = NumericFeature("bla")(1.0)
    assert(vocabWeights(newFtr) == initVal)
    val hashWeights = HashWeights(new HashIndexer(10), initVal)
    assert(hashWeights(newFtr) == initVal)
  }

  test("hash weights should collide") {
    val w = HashWeights(new HashIndexer(1))
    val ftr1 = NumericFeature("foo")(1.0)
    assert(w(ftr1) == 0.0)
    w(ftr1) = 7.0
    val ftr2 = CategoricFeature("bar", "bla")(1.0)
    assert(w(ftr2) == 7.0)
  }

  test("one vocab indexers can be shared by several weights") {
    val ixer = new VocabularyIndexer

    val w1 = VocabWeights(ixer, 0.0)
    val w2 = VocabWeights(ixer, 0.0)

    val ftrs = Seq("red", "green", "blue").map(color => NumericFeature(color)(1.0))
    for ((ftr, i) <- ftrs.zipWithIndex) {
      w1(ftr) = i+1
    }

    w2(ftrs(2)) = 11.0

    assert(w2(ftrs(2)) == 11.0)
  }

  test("createInitLike should work for block weights") {
    val w: Weights = BlockWeights(
      weightBlocks = Array(
        VocabWeights(new VocabularyIndexer, 1.0),
        HashWeights(new HashIndexer(10), 2.0)
      ),
      defaultWeights = VocabWeights(new VocabularyIndexer, 3.0),
      ftr2block = (ftr: Feature) => {
        Map("color" -> 0, "size" -> 1).getOrElse(ftr.name, -1)
      }
    )

    assert(w(NumericFeature("color")(1.0)) == 1.0)
    assert(w(NumericFeature("size")(1.0)) == 2.0)
    assert(w(NumericFeature("foo")(1.0)) == 3.0)

    val newWeights = Weights.createInitLike(w, 10.0)
    assert(newWeights(NumericFeature("color")(1.0)) == 10.0)
    assert(newWeights(NumericFeature("size")(1.0)) == 10.0)
    assert(newWeights(NumericFeature("foo")(1.0)) == 10.0)
  }

  test("block weights objects should be unique") {
    val w1 = VocabWeights(new VocabularyIndexer)
    val w2 = HashWeights(new HashIndexer(10))
    val w3 = HashWeights(new HashIndexer(20))
    assertThrows[Exception](BlockWeights(Array(w1, w2, w3), w2, (ftr: Feature) => 0))
    assertThrows[Exception](BlockWeights(Array(w1, w2, w1), w3, (ftr: Feature) => 0))
    assertThrows[Exception](BlockWeights(Array(w1), w1, (ftr: Feature) => 0))

    // this should not throw any exception
    BlockWeights(Array(w1, w2), w3, (ftr: Feature) => 0)
  }
}
