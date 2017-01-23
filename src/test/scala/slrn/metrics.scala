package slrn.metrics

import org.scalatest.FunSuite

class NormalizedEntropyTest extends FunSuite {

  test("NE of average rate prediction should be 1") {
    val ne = new NormalizedEntropy
    val data = List(
      (1, 0.5),
      (0, 0.5),
      (1, 0.5),
      (0, 0.5),
      (0, 0.5),
      (1, 0.5)
    )
    for ((y, p) <- data) {
      ne.add(y, p)
    }

    assert(math.abs(1.0 - ne.apply()) < 1e-6)
  }

  test("NE of perfect predition should be 0") {
    val ne = new NormalizedEntropy
    val data = List(
      (1, 1.0),
      (0, 0.0),
      (1, 1.0),
      (0, 0.0),
      (0, 0.0),
      (1, 1.0)
    )
    for ((y, p) <- data) {
      ne.add(y, p)
    }

    assert(math.abs(0.0 - ne.apply) < 1e-6)
  }

  test("NE of completely wrong prediction should be high") {
    val ne = new NormalizedEntropy
    val data = List(
      (1, 0.1),
      (0, 0.9),
      (1, 0.05),
      (0, 0.8),
      (0, 0.98),
      (1, 0.2)
    )
    for ((y, p) <- data) {
      ne.add(y, p)
    }

    assert(ne.apply > 1.0)
  }

  test("NE of good prediction should be low") {
    val ne = new NormalizedEntropy
    val data = List(
      (1, 0.9),
      (0, 0.1),
      (1, 0.85),
      (0, 0.05),
      (0, 0.2),
      (1, 0.95)
    )
    for ((y, p) <- data) {
      ne.add(y, p)
    }

    assert(ne.apply < 1.0)
  }

  test("NE of good but uncalibrated prediction should be high") {
    val ne = new NormalizedEntropy
    val data = List(
      (1, 0.9),
      (0, 0.8),
      (1, 0.9),
      (0, 0.8),
      (0, 0.8),
      (1, 0.9)
    )
    for ((y, p) <- data) {
      ne.add(y, p)
    }

    assert(ne.apply > 1)
  }

  test("NE should not crash when predictions are very close to 0 or 1") {
    val ne = new NormalizedEntropy
    val data = List(
      (1, 1.0),
      (0, 0.0),
      (1, 0.999990000),
      (0, 0.000000001),
      (0, 0.000000001),
      (1, 0.999999999)
    )
    for ((y, p) <- data) {
      ne.add(y, p)
    }
  }

}

class RootMeanSquareErrorTest extends FunSuite {

  test("RMSE should be initially 0") {
    val rmse = new RootMeanSquareError
    assert(rmse.apply == 0.0)
  }

  test("RMSE of perfect prediction should be 0") {
    val rmse = new RootMeanSquareError
    val data = List(
      (10.0, 10.0),
      (0.0, 0.0),
      (2.0, 2.0),
      (-7.0, -7.0)
    )
    for ((y, p) <- data) {
      rmse.add(y, p)
    }
    assert(math.abs(rmse.apply - 0.0) < 1e-6)
  }

  test("RMSE should give the correct result") {
    val rmse = new RootMeanSquareError
    val data = List(
      (1.0, 1.0),
      (2.0, 3.0),
      (-5.0, -3.0),
      (6.0, 3.0)
    )
    for ((y, p) <- data) {
      rmse.add(y, p)
    }
    assert(math.abs(rmse.apply - 1.87082869339) < 1e-6)
  }
}
