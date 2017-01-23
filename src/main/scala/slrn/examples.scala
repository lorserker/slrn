package slrn.examples

import java.io.PrintWriter

import slrn.feature.{ContinuousFeature, DiscreteFeature, Feature}
import slrn.metrics.{NormalizedEntropy, RootMeanSquareError}
import slrn.model.{LinearPrediction, LogisticPrediction, Model}
import slrn.transform.{ApplyAll, Scaler, UnitLength}
import slrn.weights._

object AirlineDelayClassificationExample {

  def main(args: Array[String]): Unit = {
    val predictionLogFNm = args(0)
    val pw = new PrintWriter(predictionLogFNm)

//    val defaultWeights = VocabWeights(new VocabularyIndexer)
//    val hashWeights = HashWeights(new HashIndexer(10000))
//
//    val model = new BlockWeights(
//      Array[Weights](hashWeights),
//      defaultWeights,
//      (ftr: Feature) => if (ftr.name == "orig-dest") 0 else -1) with LogisticPrediction

    val model = Model.classification()

    //val learner = new slrn.model.ConstantStepSGD(learningRate=0.01, model=model)
    val learner = new slrn.model.LocalVarSGD(model)

    val metric = new NormalizedEntropy

    val scale = new Scaler

    for ((y, rawFtrs) <- Data.exampleIterator()) {
      val target = if (y > 60) 1.0 else 0.0
      val ftrs = scale(rawFtrs)

      val p = model.predict(ftrs)

      metric.add(target, p)

      pw.println(s"$target\t$p\t${metric()}")

      learner.learn(target, ftrs)
    }

    pw.close()
  }
}

object AirlineDelayRegressionExample {

  def main(args: Array[String]): Unit = {
    val predictionLogFNm = args(0)
    val pw = new PrintWriter(predictionLogFNm)

//    val defaultWeights = VocabWeights(new VocabularyIndexer)
//    val hashWeights = HashWeights(new HashIndexer(10000))
//
//    val model = new BlockWeights(
//      Array[Weights](hashWeights),
//      defaultWeights,
//      (ftr: Feature) => if (ftr.name == "orig-dest") 0 else -1) with LinearPrediction

    val model = Model.regression()

    //val learner = new slrn.model.ConstantStepSGD(learningRate=0.01, model=model)
    val learner = new slrn.model.LocalVarSGD(model)

    val metric = new RootMeanSquareError

    val scale = new Scaler

    for ((target, rawFtrs) <- Data.exampleIterator()) {
      val ftrs = scale(rawFtrs)

      val p = model.predict(ftrs)

      metric.add(target, p)

      pw.println(s"$target\t$p\t${metric()}")

      learner.learn(target, ftrs)
    }

    pw.close()
  }
}

object Data {
  def exampleIterator(): Iterator[(Double, Set[Feature])] = {
    val lineIterator = io.Source.fromFile("datasets/airline.csv").getLines

    for (line <- lineIterator.drop(1)) yield {
      val cols = line.trim.split(",")
      val target = cols(0).toDouble
      val orig = cols(1)
      val dest = cols(2)
      val distance = cols(3).toDouble
      val carrier = cols(4)
      val flightNum = carrier + cols(5)
      val departureTime = cols(6).toDouble
      val depDelay = cols(7).toDouble
      val monthDate = s"${cols(8)}-${cols(9)}"
      val dayOfWeek = cols(10)

      (target, Set[Feature](
        DiscreteFeature("orig", orig)(),
        DiscreteFeature("dest", dest)(),
        Feature.cross(
          DiscreteFeature("orig", orig)(),
          DiscreteFeature("dest", dest)()
        ),
        ContinuousFeature("distance")(distance),
        ContinuousFeature("depart")(departureTime),
        DiscreteFeature("carrier", carrier)(),
        DiscreteFeature("flight", flightNum)(),
        DiscreteFeature("mdate", monthDate)(),
        DiscreteFeature("dow", dayOfWeek)(),
        ContinuousFeature("dep-delay")(depDelay),
        Feature.bias
      ))
    }
  }
}
