# slrn

[![CircleCI](https://circleci.com/gh/lorserker/slrn.svg?style=svg)](https://circleci.com/gh/lorserker/slrn)

slrn aims to be a simple, easy to use, minimalistic machine learning library that supports the most common use-cases like: classification, regression, online learning out of the box and exposes the functionality in simple interfaces. slrn tries to have few dependencies, few configuration parameters, and implement simple optimization algorithms that are fast to run.

The target audience for slrn are smaller teams of engineers which don't necessarily have a lot of machine learning experience, but need to build some machine learning functionality into their projects quickly. The hope is that slrn can offer a good, low-maintenance and low-cost initial solution for projects whith a machine learning component.

In the following sections we first introduce the basic building blocks like: features, prediction models, weights, learners, etc. And finally we show an example of modeling airline delays using regression and classification.

## Feature Representation

slrn supports two kinds of features: `ContinuousFeature` and `DiscreteFeature`

```scala
case class ContinuousFeature(name: String)(val value: Double) extends Feature

case class DiscreteFeature(name: String, nominal: String)(val value: Double = 1.0) extends Feature
```

Continuous features are used to represent numeric values like distance and temperature

```scala
val distance = ContinuousFeature("distance")(100.0)
```

Discrete features are used for categoric values like color, shape, gender, etc.

```scala
val color = DiscreteFeature("color", "blue")()
```

It is possible to combine several discrete features by 'crossing' them. This way we can represent for example the various properties of a coffee in one feature:

```scala
val coffee = Feature.cross(
  DiscreteFeature("type", "cappucino"),
  DiscreteFeature("size", "small"),
  DiscreteFeature("milk", "lowfat"),
  DiscreteFeature("caffeine", "decaf")
)
```

### Feature Vectors
Feature vectors are implemented simply as `Set[Feature]`. This way the user of the library doesn't have to know or keep track of any of the following: how many dimensions the feature space has, what index (coordinate) a particular feature has in the feature vector. 

## Prediction Models
In general a prediction model is an object of type ```Weights with Prediction```. There are several implementations of the ```Weights``` trait and the ```Prediction``` trait.

A prediction model can make used to make a prediction for a set of features `Set[Feature]` like this:

```scala
val prediction = model.predict(ftrs)
```

To create a model, easiest is to just use the helper functions in the ```Model``` object:

```scala
 Model.regression()
 Model.classification()
```
or the versions with a parameter ```n``` that specifies the maximum number of weights that will be used:

```scala
Model.regression(n)
Model.classification(n)
```
Specifying the maximum number of weights ```n``` is useful if there are a large number of features in sparse models or if a lot of combinations like the coffee example above are used and we are in danger of the dimensionality exploding too much. Under the hood ```HashWeights``` and ```HashIndexer``` are used to hash each feature to one of ```n``` buckets.

For a detailed specification of how model weights are represented we can use ```BlockWeights```
```scala
val blocks = Array[Weights](
  new VocabWeights(new VocabularyIndexer),
  new HashWeights(new HashIndexer(20000)),
  new HashWeights(new HashIndexer(100000))
)
val defaultWeights = new new HashWeights(new HashIndexer(100000))
val ftr2blockFn = (ftr: Feature) => Map("country" -> 0, "city" -> 1, "zip" -> 2).getOrElse(ftr.name, -1)

val model = new BlockWeights(blocks, defaultWeights, ftr2blockFn) with LogisticPrediction
```
In the above example we have three weight blocks and one block of default weights. The function ```ftr2blockFn``` specified that weights for country features will be stored in the first block that is a ```VocabWeights``` with no upper limit on dimensions. Weights for cities will be stored in the second block that is implemented by a ```HashWeights``` with 20000 dimensions. Zip codes will be stored in the third block. All other features will have weights in ```defaultWeights```. The last line in the code snippet creates the model as an instance of ```BlockWeights``` with a mixin of the ```LogisticPrediction``` trait in order to have the logistic function as the activation function for the prediction.

## Learners
Learners have the role of adjusting the weights for a model when being shown training examples.

Training examples are shown to the learner one by one, so online learning on a data stream is supported.

A training example is just a pair of a target value and a set of features ```(Double, Set[Feature])```

We train a model through a learner like this:

```scala
learner.learn(target, ftrs)
```
Roghly what happens here is that the model is used to make a prediction for `ftrs`, then the prediction is compared to `target`, and an adjustment of the relavant weights is made according to the prediction error.

slrn supports several types of learners like `ConstantStepSGD` and `LocalVarSGD`. `LocalVarSGD` is special because it does not have any meta-parameters, so to train a classification model with zero configuration you can simply do something like this:

```scala
val model = Model.classification()
val learner = LocalVarSGD(model)

for ((target, ftrs) <- trainingExamples) {
  learner.learn(target, ftrs)
}
```
and then use the model to predict like this
```scala
for (ftrs <- examples) {
  val prediction = model.predict(ftrs)
}
```

## Example - Airline Delay Prediction
In this example we'll use a regression model to predict how many minutes of delay an airplane will have.

We are using a sample dataset from kaggle which looks like this:

| ArrDelay | Origin | Dest | Distance | UniqueCarrier | FlightNum | CRSDepTime | DepDelay | Month | DayofMonth | DayOfWeek |
| -------- | ------ | ---- | -------- | ------------- | --------- | ---------- | -------- | ----- | ---------- | --------- |
|      -14 | IAD    | TPA  |      810 | WN            |       335 |       1955 |        8 |  True |          3 |         4 |
|        2 | IAD    | TPA  |      810 | WN            |      3231 |        735 |       19 |  True |          3 |         4 |
|       14 | IND    | BWI  |      515 | WN            |       448 |        620 |        8 |  True |          3 |         4 |
|       34 | IND    | BWI  |      515 | WN            |      3920 |       1755 |       34 |  True |          3 |         4 |
|       11 | IND    | JAX  |      688 | WN            |       378 |       1915 |       25 |  True |          3 |         4 |
|       57 | IND    | LAS  |     1591 | WN            |       509 |       1830 |       67 |  True |          3 |         4 |
|        1 | IND    | MCO  |      828 | WN            |       100 |        700 |        6 |  True |          3 |         4 |
|       80 | IND    | MCO  |      828 | WN            |      1333 |       1510 |       94 |  True |          3 |         4 |
|       11 | IND    | MDW  |      162 | WN            |      2272 |       1020 |        9 |  True |          3 |         4 |

We want to model the arrival delay `ArrDelay` by using the other features given in the above table.

```scala
val model = Model.regression()
val learner = new LocalVarSGD(model)
val metric = new RootMeanSquareError
val scale = new Scaler

for ((target, rawFtrs) <- Data.exampleIterator()) {
  val ftrs = scale(rawFtrs)

  val p = model.predict(ftrs)

  metric.add(target, p)

  learner.learn(target, ftrs)
}
```
Iterating through the examples we do the following steps:
- we scale the features using a `Scaler`
- we make a prediction using the `model`
- we update the performance metric (which in this example is `RootMeanSquareError`)
- we adjust the model using the `learner`

The training examples could be generated like this:
```scala
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
      DiscreteFeature("orig", orig)(),
      DiscreteFeature("dest", dest)(),
      Feature.cross(
        DiscreteFeature("orig", orig)(),
        DiscreteFeature("dest", dest)()
      ),
      ContinuousFeature("distance")(distance),
      ContinuousFeature("depart")(departureTime),
      DiscreteFeature("carrier", carrier)(),
      DiscreteFeature("flight", flightNum)(),
      DiscreteFeature("mdate", monthDate)(),
      DiscreteFeature("dow", dayOfWeek)(),
      ContinuousFeature("dep-delay")(depDelay),
      Feature.bias
    ))
  }
}
```

If we now wanted to change our setup and whether the plane will be late more than an hour (instead of predicting exactly how many minutes it will be late) we have to make just three small changes in the example above.

First of all, the model should be a classification model, so:
```scala
val model = Model.classification()
```
Secondly, the target is 1 if the delay is larger than one hour and 0 otherwise:
```scala
val target = if (delay > 60) 1.0 else 0.0
```
And finally, we change the performance metric to `NormalizedEntropy`
```scala
val metric = new NormalizedEntropy
```

### Running the Examples

In order to run the airline delay examples you can type the commands:
```
sbt "run-main slrn.examples.AirlineDelayRegressionExample logfile.log"
```
```
sbt "run-main slrn.examples.AirlineDelayClassificationExample logfile.log"
```




