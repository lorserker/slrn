package slrn.transform

import slrn.transform._
import org.scalatest.FunSuite

import scala.util.{Failure, Try}

class TransformTest extends FunSuite {

  test("OnlineMeanStd should work properly") {
    val onlineMeanStd = new OnlineMeanStd

    assert(onlineMeanStd.n == 0)
    assert(onlineMeanStd.mean == 0)
    assert(Try(onlineMeanStd.variance).isFailure)

    val v = util.Random.nextGaussian() + 7
    onlineMeanStd.add(v)

    assert(onlineMeanStd.n == 1)
    assert(math.abs(onlineMeanStd.mean - v) < 1e-6)
    assert(Try(onlineMeanStd.variance).isFailure)

    (1 until 1000).foreach { i =>
      onlineMeanStd.add(util.Random.nextGaussian() + 7)
    }

    assert(onlineMeanStd.n == 1000)
    assert(math.abs(onlineMeanStd.mean - 7.0) < 0.1)
    assert(math.abs(Try(onlineMeanStd.variance).get - 1.0) < 0.1)
  }
}