package slrn

import slrn.feature._
import scala.io.Source

package object io {

  def examples(source: Source): Iterator[(Double, Set[Feature])] = source.getLines().map(parseLine)

  private def parseLine(line: String): (Double, Set[Feature]) = {
    val cols = line.split("\t").map(_.trim)
    require(cols.length > 0)
    val label = cols(0).toInt
    val ftrs = cols.drop(1).map {
      case s if s.startsWith("n:") => {
        val (ftrname, ftrval) = splitFeature(s.drop(2))
        ContinuousFeature(name = ftrname)(value = ftrval.toDouble)
      }
      case s if s.startsWith("c:") => {
        val (ftrname, ftrval) = splitFeature(s.drop(2))
        DiscreteFeature(name = ftrname, nominal = ftrval)(value = 1.0)
      }
    }.toSet[Feature]
    (label, ftrs)
  }

  private def splitFeature(s: String): (String, String) = {
    val i = s.indexOf('=')
    (s.take(i), s.drop(i+1))
  }

  object Mushroom {

    def examples(source: Source): Iterator[(Double, Set[Feature])] = {
      val lines = source.getLines().toVector
      val header = lines(0).trim().split(",").map(_.trim)

      val rows = for (line <- lines.drop(1)) yield {
        val cols = line.trim.split(",").map(_.trim)
        val target = Map("p" -> 1.0, "e" -> 0.0)(cols(0))
        val ftrs = header.zip(cols).drop(1).map{ case (h, c) => DiscreteFeature(name=h, nominal=c)(1.0) }.toSet[Feature]
        (target, ftrs)
      }

      rows.iterator
    }
  }

  object CreditCard {

    def examples(source: Source): Iterator[(Double, Set[Feature])] = {
      val lineIterator = source.getLines()
      val header = lineIterator.next().trim().split(",").map(_.trim)
      for ((line, i) <- lineIterator.zipWithIndex) yield {
        val cols = line.trim.split(",").map(_.trim)
        val target = Map("\"0\"" -> 0.0, "\"1\"" -> 1.0)(cols.last)
        val ftrs = for ((col, i) <- cols.drop(1).dropRight(1).zipWithIndex) yield {
          ContinuousFeature(header(i+1))(col.toDouble)
        }
        (target, ftrs.toSet[Feature])
      }
    }

  }

}


