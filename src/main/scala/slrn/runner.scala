package slrn.runner

import java.io.{FileOutputStream, PrintWriter}

import slrn.feature.CategoricFeature
import slrn.model.{AdaDelta, MiniBatch}

object RunSimpleSGD {

  def main(args: Array[String]): Unit = {

    val outModelFnm = "weights.tsv"
    val pw = new PrintWriter("prediction_raw.log")

    val debugWriter = new PrintWriter("debug.tsv")

    val model = new slrn.model.DictionaryWeights()
    //val learner = new slrn.model.ConstantStepSGD(learningRate=0.03, model=model)
    //val learner = new MiniBatch(10, new slrn.model.ConstantStepSGD(learningRate=0.1, model=model))
    val learner = new slrn.model.LocalVarSGD(model)
    //val learner = new MiniBatch(10, new slrn.model.LocalVarSGD(model))
    //val learner = new AdaDelta(model)

    for (((label, ftrs), i) <- slrn.io.examples(scala.io.Source.stdin).zipWithIndex) {
      if (i % 100000 == 0) {
        println(i)
      }

      //if (i > 100) return

      val segmentFtrArray = ftrs.filter(_.name == "cust_appd").toArray
      val segment = segmentFtrArray(0).asInstanceOf[CategoricFeature].nominal

      val p = model.predict(ftrs)

      pw.println(s"$label\t${p}\t${segment}")

//      if (segment == "ec#1" && label == 0) {
//        if (math.random < 0.1) {
//          learner.learn(label, ftrs)
//        }
//      } else {
//        learner.learn(label, ftrs)
//      }

      learner.learn(label, ftrs)

//      for (ftr <- ftrs) {
//
//        val row = List(
//          label.toString,
//          p.toString,
//          ftr.asInstanceOf[CategoricFeature].nominal,
//          learner.gStats.avgGradient(ftr).toString,
//          learner.gStats.avgSquareGradient(ftr).toString,
//          learner.gStats.avgDiagHess(ftr).toString,
//          learner.gStats.memory(ftr).toString,
//          learner.gStats.learningRate(ftr).toString
//        )
//        debugWriter.println(row.mkString("\t"))
//      }
    }

    pw.close()
    debugWriter.close()

    model.save(new FileOutputStream(outModelFnm))

  }

}