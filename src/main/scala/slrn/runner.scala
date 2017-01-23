package slrn.runner

import java.io.{FileOutputStream, PrintWriter}

import slrn.feature.{DiscreteFeature, Feature}
import slrn.metrics.NormalizedEntropy
import slrn.weights._
import slrn.model.{ConstantStepSGD, LocalVarSGD, LogisticPrediction, MiniBatch}
import slrn.sampling.{PriorCorrectedWeights, Sampler}
import slrn.transform.{ApplyAll, Scaler, UnitLength}

object RunSimpleSGD {

  def main(args: Array[String]): Unit = {
    val predictionLogFNm = args(0)
    println(predictionLogFNm)

    val outModelFnm = "weights.tsv"
    val pw = new PrintWriter(predictionLogFNm)

    val appdWeights = new VocabWeights(new VocabularyIndexer)
    val brandWeights = new VocabWeights(new VocabularyIndexer)
    val defaultWeights = new HashWeights(new HashIndexer(100000))

    def ftr2block(ftr: Feature): Int = Map("cust_appd" -> 0, "brand" -> 1).getOrElse(ftr.name, -1)

    //val model = new BlockWeights(Array(appdWeights, brandWeights), defaultWeights, ftr2block) with LogisticPrediction

    //val model = new slrn.model.DictionaryWeights()
    val model = new VocabWeights(new VocabularyIndexer) with LogisticPrediction
    //val model = new HashWeights(new HashIndexer(100000)) with LogisticPrediction
    //val learner = new slrn.model.ConstantStepSGD(learningRate=0.03, model=model)
    //val learner = new MiniBatch(10, new slrn.model.ConstantStepSGD(learningRate=0.1, model=model))
    val learner = new slrn.model.LocalVarSGD(model)
    //val learner = new MiniBatch(10, new slrn.model.LocalVarSGD(model))

    //val sampler = new Sampler(targetRate = 0.2, (ftrs: Set[Feature]) => ftrs.filter(_.name == "cust_appd").toSeq.head, 1e-3)

    val normEnt = new NormalizedEntropy

    for (((label, ftrs), i) <- slrn.io.examples(scala.io.Source.stdin).zipWithIndex) {
      if (i % 100000 == 0) {
        println(i)
      }

      val segmentFtrArray = ftrs.filter(_.name == "cust_appd").toArray
      val segment = segmentFtrArray(0).asInstanceOf[DiscreteFeature].nominal

      val p = model.predict(ftrs)
      //val p = (new PriorCorrectedWeights(weights=model, sampler=sampler) with LogisticPrediction).predict(ftrs)

      normEnt.add(label, p)

      pw.println(s"$label\t${p}\t${normEnt.apply}\t${segment}")

      if (true) {
      //if (sampler((label, ftrs))) {
        learner.learn(label, ftrs)
      }

    }

    //println(sampler.groupStats)

    pw.close()
  }
}

object MushroomsExample {
  def main(args: Array[String]): Unit = {
    val pw = new PrintWriter("prediction.log")

    val model = new VocabWeights(new VocabularyIndexer) with LogisticPrediction
    //val model = new HashWeights(new HashIndexer(1000)) with LogisticPrediction
    //val learner = new ConstantStepSGD(learningRate = 0.1, model=model)
    //val learner = new MiniBatch(5, new LocalVarSGD(model))
    //val learner = new MiniBatch(5, new ConstantStepSGD(learningRate = 0.1, model=model))
    val learner = new LocalVarSGD(model)

    util.Random.setSeed(1234L)
    val examples = util.Random.shuffle(slrn.io.Mushroom.examples(io.Source.fromFile("datasets/mushrooms.csv")).toVector)

    for (((target, ftrs), i) <- examples.zipWithIndex) {
      if (i > 5000) {
        val p = model.predict(ftrs)
        pw.println(s"$target\t${p}")
      } else {
        learner.learn(target, ftrs)
      }
    }

    pw.close()
  }
}

object CreditCardExample {
  def main(args: Array[String]): Unit = {
    util.Random.setSeed(1234L)
    val pw = new PrintWriter("prediction.log")

    val model = new VocabWeights(new VocabularyIndexer) with LogisticPrediction

    val learner = new ConstantStepSGD(learningRate = 0.1, model=model)
    //val learner = new LocalVarSGD(model)

    val scaler = new Scaler
    for (((target, rawFtrs), i) <- slrn.io.CreditCard.examples(io.Source.fromFile("datasets/creditcard.csv")).zipWithIndex) {
      val ftrs = ApplyAll(List(scaler, UnitLength))(rawFtrs)

      if (i > 5000) {
        val p = model.predict(ftrs)
        pw.println(s"$target\t${p}")
      } else {
        if (target > 0.0 || util.Random.nextDouble() < 0.05)
        learner.learn(target, ftrs)
      }
    }

    pw.close()
  }
}