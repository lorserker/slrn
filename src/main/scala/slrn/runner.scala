package slrn.runner

import java.io.{FileOutputStream, PrintWriter}

import slrn.feature.{CategoricFeature, Feature}
import slrn.weights._
import slrn.model.{ConstantStepSGD, LocalVarSGD, LogisticPrediction, MiniBatch}
import slrn.transform.{ApplyAll, Scaler, UnitVector}

object RunSimpleSGD {

  def main(args: Array[String]): Unit = {

    val outModelFnm = "weights.tsv"
    val pw = new PrintWriter("prediction_raw.log")

    val debugWriter = new PrintWriter("debug.tsv")

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

    for (((label, ftrs), i) <- slrn.io.examples(scala.io.Source.stdin).zipWithIndex) {
      if (i % 100000 == 0) {
        println(i)
      }

      val segmentFtrArray = ftrs.filter(_.name == "cust_appd").toArray
      val segment = segmentFtrArray(0).asInstanceOf[CategoricFeature].nominal

      val p = model.predict(ftrs)

      pw.println(s"$label\t${p}\t${segment}")

      learner.learn(label, ftrs)

    }

    pw.close()
    debugWriter.close()
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

    //val learner = new ConstantStepSGD(learningRate = 0.1, model=model)
    val learner = new LocalVarSGD(model)

    val scaler = new Scaler
    for (((target, rawFtrs), i) <- slrn.io.CreditCard.examples(io.Source.fromFile("datasets/creditcard.csv")).zipWithIndex) {
      val ftrs = ApplyAll(List(scaler, UnitVector))(rawFtrs)

      if (i > 150000) {
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