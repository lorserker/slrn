package slrn.feature

import org.scalatest.FunSuite


class FeaturesTest extends FunSuite {
  test("feature combinations should work") {
    val color = DiscreteFeature("color", "green")(2.0)
    val size = DiscreteFeature("size", "XXL")(3.0)

    val c1 = Feature.cross(color, size)
    assert(c1.name == "color|size")
    assert(c1.nominal == "green|XXL")
    assert(c1.value == 6.0)

    val c2 = Feature.cross("combination", color, size)
    assert(c2.name == "combination")
    assert(c2.nominal == "green|XXL")
    assert(c2.value == 6.0)
  }
}
