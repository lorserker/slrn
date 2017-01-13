package slrn.examples

import java.io.PrintWriter

import slrn.feature.{ContinuousFeature, DiscreteFeature, Feature}
import slrn.metrics.{NormalizedEntropy, RootMeanSquareError}
import slrn.model.{DotPrediction, LogisticPrediction}
import slrn.transform.Scaler
import slrn.weights._

object AirlineDelayClassificationExample {

  def main(args: Array[String]): Unit = {
    val predictionLogFNm = args(0)
    val pw = new PrintWriter(predictionLogFNm)

    val defaultWeights = VocabWeights(new VocabularyIndexer)
    val hashWeights = HashWeights(new HashIndexer(10000))

    val model = new BlockWeights(
      Array[Weights](hashWeights),
      defaultWeights,
      (ftr: Feature) => if (ftr.name == "orig-dest") 0 else -1) with LogisticPrediction

    //val learner = new slrn.model.ConstantStepSGD(learningRate=0.01, model=model)
    val learner = new slrn.model.LocalVarSGD(model)

    val metric = new NormalizedEntropy

    val scale = new Scaler

    for ((y, unscaledFtrs) <- Data.exampleIterator()) {
      val target = if (y > 60) 1.0 else 0.0
      val ftrs = scale(unscaledFtrs)

      val p = model.predict(ftrs)

      metric.add(target, p)

      pw.println(s"$target\t$p\t${metric.get}")

      learner.learn(target, ftrs)
    }

    pw.close()
  }
}

object AirlineDelayRegressionExample {

  def main(args: Array[String]): Unit = {
    val predictionLogFNm = args(0)
    val pw = new PrintWriter(predictionLogFNm)

    val defaultWeights = VocabWeights(new VocabularyIndexer)
    val hashWeights = HashWeights(new HashIndexer(10000))

    val model = new BlockWeights(
      Array[Weights](hashWeights),
      defaultWeights,
      (ftr: Feature) => if (ftr.name == "orig-dest") 0 else -1) with DotPrediction

    //val learner = new slrn.model.ConstantStepSGD(learningRate=0.01, model=model)
    val learner = new slrn.model.LocalVarSGD(model)

    val metric = new RootMeanSquareError

    val scale = new Scaler

    for ((target, unscaledFtrs) <- Data.exampleIterator()) {
      val ftrs = scale(unscaledFtrs)

      val p = model.predict(ftrs)

      metric.add(target, p)

      pw.println(s"$target\t$p\t${metric.get}")

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
        DiscreteFeature(name="orig", nominal=orig)(),
        DiscreteFeature(name="dest", nominal=dest)(),
        Feature.combine(DiscreteFeature(name="orig", nominal=orig)(), DiscreteFeature(name="dest", nominal=dest)()),
        ContinuousFeature(name="distance")(value=distance),
        ContinuousFeature(name="depart")(value=departureTime),
        DiscreteFeature(name="carrier", nominal=carrier)(),
        DiscreteFeature(name="flight", nominal=flightNum)(),
        DiscreteFeature(name="mdate", nominal=monthDate)(),
        DiscreteFeature(name="dow", nominal=dayOfWeek)(),
        ContinuousFeature(name="dep-delay")(value=depDelay),
        Feature.bias
      ))
    }
  }
}
