package slrn

import slrn.feature._

import scala.io.Source

package object io {

  def examples(source: Source): Iterator[(Int, Set[Feature])] = source.getLines().map(parseLine)

  private def parseLine(line: String): (Int, Set[Feature]) = {
    val cols = line.split("\t").map(_.trim)
    require(cols.length > 0)
    val label = cols(0).toInt
    val ftrs = cols.drop(1).map {
      case s if s.startsWith("n:") => {
        val (ftrname, ftrval) = splitFeature(s.drop(2))
        NumericFeature(name = ftrname)(value = ftrval.toDouble)
      }
      case s if s.startsWith("c:") => {
        val (ftrname, ftrval) = splitFeature(s.drop(2))
        CategoricFeature(name = ftrname, nominal = ftrval)(value = 1.0)
      }
    }.toSet[Feature]
    (label, ftrs)
  }

  private def splitFeature(s: String): (String, String) = {
    val i = s.indexOf('=')
    (s.take(i), s.drop(i+1))
  }
}


