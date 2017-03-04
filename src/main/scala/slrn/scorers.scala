package slrn.scorers

import java.io.PrintWriter

import slrn.feature._
import slrn.model._
import slrn.weights._

object ScorerRunner {
  def main(args: Array[String]): Unit = {
    //val fnm = "/home/ldali/okr/training_time/data/trdata_sorted_2.tsv"
    val fnm = args(0)
    val scorerName = args(1)
    val logFnm = args(2)

    val pw = new PrintWriter(logFnm)

    //val model = new VocabWeights(new VocabularyIndexer) with LogisticPrediction
    val model = Model.classification()

    val learner = new ConstantStepSGD(learningRate = 0.03, model = model)
    //val learner = new LocalVarSGD(model)

    for ((x, i) <- Data.examples(fnm, Data.alwaysTrue, Data.scorerFtrex(scorerName)).zipWithIndex) {
      if (i % 100000 == 0) println(i)

      val p = model.predict(x.ftrs)

      pw.println(s"${x.target}\t$p\t${x.ec}\t${x.order}")

      learner.learn(x.target, x.ftrs)
    }

    pw.close()
  }
}

object ScorerTestRunner {
  def main(args: Array[String]): Unit = {
    val trainfnm = args(0)
    val testfnm = args(1)
    val scorerName = args(2)
    val logFnm = args(3)

    val pw = new PrintWriter(logFnm)

    val model = Model.classification()

    val learner = new ConstantStepSGD(learningRate = 0.03, model = model)

    for ((x, i) <- Data.examples(trainfnm, Data.alwaysTrue, Data.scorerFtrex(scorerName)).zipWithIndex) {
      if (i % 100000 == 0) println("train", i)

      val p = model.predict(x.ftrs)

      //pw.println(s"${x.target}\t$p\t${x.ec}\t${x.order}")

      learner.learn(x.target, x.ftrs)
    }

    for ((x, i) <- Data.examples(testfnm, Data.alwaysTrue, Data.scorerFtrex(scorerName)).zipWithIndex) {
      if (i % 100000 == 0) println("test", i)

      val p = model.predict(x.ftrs)

      pw.println(s"${x.target}\t$p\t${x.ec}\t${x.order}")
    }

    pw.close()
  }
}

case class Example(order: String, ec: Boolean, target: Double, ftrs: Set[Feature])

object Data {

  val scorerFtrex = Map(
    "zip" -> zipLevels _,
    "zipall" -> zipLevelsAll _,
    "dom" -> domain _,
    "art" -> articles _,
    "brandart" -> brandArticles _,
    "how" -> hourOfWeek _,
    "brand" -> brands _
  )

  def examples(fnm: String, p: Array[String] => Boolean, ftrex: Array[String] => Set[Feature]) = {
    rowIterator(fnm).filter(p).map(row => Example(order(row), isEC(row), label(row), ec(row) union ftrex(row)))
  }

  def alwaysTrue = (row: Array[String]) => true

  def label(cols: Array[String]): Double = cols(0).toDouble

  def order(cols: Array[String]): String = cols(1)

  def isEC(cols: Array[String]): Boolean = cols(3) == "true"

  def ec(cols: Array[String]): Set[Feature] = Set(DiscreteFeature("ec", cols(3))())

  def zip(cols: Array[String]): Set[Feature] = Set(DiscreteFeature("zip", cols(5))())

  def zipLevels(cols: Array[String]): Set[Feature] = {
    cols(5) match {
      case z if z.startsWith("DE") || z.startsWith("AT") =>
        Set[Feature](
          DiscreteFeature("zip_1", z.take(4))(),
          DiscreteFeature("zip_2", z.take(5))()
        )
      case z if z.startsWith("CH") =>
        Set[Feature](
          DiscreteFeature("zip_1", z.take(4))(),
          DiscreteFeature("zip_2", z.take(6))()
        )
    }
  }

  def zipLevelsAll(cols: Array[String]): Set[Feature] = zipLevels(cols) union zip(cols)

  def hourOfWeek(cols: Array[String]): Set[Feature] = Set(DiscreteFeature("HoW", cols(4))())

  def domain(cols: Array[String]): Set[Feature] = {
    val emailDomain = cols(6).split("@")(1)
    Set(DiscreteFeature("domain", emailDomain)())
  }

  def brands(cols: Array[String]): Set[Feature] = {
    skuKVs(cols).
      filter{ case (k, v) => k.length == 3 }.
      map{ case (k, v) => DiscreteFeature("brand", k)(v)}.toSet[Feature]
  }

  def articles(cols: Array[String]): Set[Feature] = {
    skuKVs(cols).
      filter{ case (k, v) => k.length == 9 }.
      map{ case (k, v) => DiscreteFeature("brand", k)(v)}.toSet[Feature]
  }

  def brandArticles(cols: Array[String]): Set[Feature] = {
    skuKVs(cols).
      map{ case (k, v) => DiscreteFeature("brand", k)(v)}.toSet[Feature]
  }

  def skuKVs(cols: Array[String]): Seq[(String, Double)] = {
    val keys = cols.drop(7).zipWithIndex.filter{ case (x, i) => i % 2 == 0 }.map(_._1)
    val values = cols.drop(7).zipWithIndex.filter{ case (x, i) => i % 2 == 1 }.map(_._1.toDouble)

    keys zip values
  }

  def rowIterator(fnm: String): Iterator[Array[String]] = {
    io.Source.fromFile(fnm).getLines().map(_.split("\t"))
  }
}
