package slrn.feature


trait Feature  extends Product {
  def name: String
  def value: Double
}

case class NumericFeature(name: String)(val value: Double) extends Feature

case class CategoricFeature(name: String, nominal: String)(val value: Double = 1.0) extends Feature


object Feature {
  def withValue(ftr: Feature, v: Double): Feature = {
    ftr match {
      case NumericFeature(name) => NumericFeature(name)(v)
      case CategoricFeature(name, nominal) => CategoricFeature(name, nominal)(v)
    }
  }
}