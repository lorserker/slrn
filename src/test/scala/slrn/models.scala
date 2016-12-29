package slrn.model

import slrn.feature._
import org.scalatest.FunSuite
import slrn.weights.{VocabWeights, VocabularyIndexer}


class ModelTest extends FunSuite {

  val epsilon = 1e-9

  val ftr1 = NumericFeature("num1")(2.0)
  val ftr2 = NumericFeature("num2")(-1.0)
  val red = CategoricFeature("color", "red")(1.0)
  val green = CategoricFeature("color", "green")(1.0)
  val blue = CategoricFeature("color", "blue")(1.0)
  val bad = CategoricFeature("signal", "red")(1.0)
  val good = CategoricFeature("signal", "green")(1.0)

  val m = new VocabWeights(new VocabularyIndexer) with LogisticPrediction
  m(ftr1) = 2.0
  m(red) = 1.0
  m(green) = -5.0
  m(bad) = -10.0
  m(good) = 7.0

  test("dot product should work properly") {
    assert(math.abs(m(ftr1) - 2.0) < epsilon)
    assert(math.abs(m(red) - 1.0) < epsilon)
    assert(math.abs(m(blue) - 0.0) < epsilon)

    assert(math.abs(m.dot(Set(ftr2, blue)) - 0.0) < epsilon)
    assert(math.abs(m.dot(Set(ftr1, green, blue)) - (-1.0)) < epsilon)
    assert(math.abs(m.dot(Set(ftr1, green, good)) - 6.0) < epsilon)
  }

  test("prediction should work properly") {
    assert(math.abs(m.predict(Set(ftr2, blue)) - 0.5) < epsilon)
    assert(math.abs(m.predict(Set(ftr1, green, blue)) - 0.2689414213699951) < epsilon)
    assert(math.abs(m.predict(Set(ftr1, green, good)) - 0.99752737684336534) < epsilon)
  }
}