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
}