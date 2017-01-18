package slrn.feature


trait Feature  extends Product {
  def name: String
  def value: Double
}

case class ContinuousFeature(name: String)(val value: Double) extends Feature

case class DiscreteFeature(name: String, nominal: String)(val value: Double = 1.0) extends Feature


object Feature {
  def withValue(ftr: Feature, v: Double): Feature = {
    ftr match {
      case ContinuousFeature(name) => ContinuousFeature(name)(v)
      case DiscreteFeature(name, nominal) => DiscreteFeature(name, nominal)(v)
    }
  }

  def bias = DiscreteFeature(name="_bias", nominal="")()

  def cross(name: String, ftrs: DiscreteFeature*): DiscreteFeature = {
    val combinedNominal = ftrs.map(_.nominal).mkString("|")
    val combinedValue = ftrs.map(_.value).product

    DiscreteFeature(name, combinedNominal)(combinedValue)
  }

  def cross(ftrs: DiscreteFeature*): DiscreteFeature = {
    val combinedName = ftrs.map(_.name).mkString("|")
    cross(combinedName, ftrs:_*)
  }
}