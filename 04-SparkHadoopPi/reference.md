# Quick Reference for PySpark

This document contains only sample codes and brief explanations.

- [Spark Overview](https://spark.apache.org/docs/latest/)
  (Quick Start, Programming Guide, etc.)
- [PySpark API documentation](https://spark.apache.org/docs/latest/api/python/)
- Hadoop 2.7.3, **Spark 2.1.0**, Python 3.4.2

## Discramer

- The following codes are NOT written, so that we can reproduce the same
  results. Some working examples can be found at 
  [this repository](https://github.com/stdiff/sandbox/tree/master/04-SparkHadoopPi).
- At your own risk.

# Spark SQL

[Spark SQL, DataFrames and Datasets Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html)

## Spark Session 

A SparkSession instance can be used for functionalities of Spark SQL

	from pyspark.sql import SparkSession
	spark = SparkSession.builder\
		                .appName("Data Analysis on Spark")\
                        .config("spark.some.config.option", "some-value")\
                        .getOrCreate() ## pyspark.sql.session.SparkSession


- appName     : as you like. Used in the Spark web UI.
- config      : a config option. We may use a SparkConf instance.
- getOrCreate : getter for the SparkSession

From now on `spark` is a SparkSession instance.

## Load Data

`spark.read` returns a
[DataFrameReader](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrameReader)
instance and we use it to load data. 

	df_batting = spark.read.csv('data/Batting.csv', header=True, nullValue='NA')

- We need `header=True` if the first line is the header.
- `parquet` and `json` can be also loaded.
- **We can specify a directory** containing multiple files.

## Output Data

`df.write` (not the spark session instance) returns a 
[DataFrameWriter](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrameWriter)
instance and we use it to write data.

	df_career.write.parquet('output_dir')

- An error arises if the specified directory exists. 
- No header line is added if we write data on CSV files. Therefore we should
  use [Apache Parquet](http://parquet.apache.org/) format.
  

## DataFrame Operations

In principle **a method returns a new DataFrame**. Therefore we have to assign
the returned DataFrame to a variable before we use the values of the new
DataFrame. 

### Look at a DataFrame

- `columns` (attr) : same as `columns` in pandas 
- `show(5)` : same as `head(5)` in pandas 
- `count()` : same as `shape[0]` in pandas 
- `printSchema()` : same as `info()` in pandas

### Tweak DataFrame

`cast()` method converts the
[data type](https://spark.apache.org/docs/latest/sql-programming-guide.html#data-types)
of a column instance. 

	df_career = df_batting.withColumn('H', df_batting.H.cast('int'))\
		                  .withColumn('AB', df_batting.AB.cast('int'))\
                          .select(['playerID','H','AB'])
    df_career.printSchema()
	## root
	##  |-- playerID: string (nullable = true)
	##  |-- H: integer (nullable = true)
	##  |-- AB: integer (nullable = true)

`int`, `float`, `string` can be used to specify. Or we may use 
`pyspark.sql.types.*`. 

	from pyspark.sql.types import IntegerType, DoubleType
	
Then a data type is given by `IntegerType()`, `DoubleType()`, `StringType()`,
`BooleanType()`, `DateType()`, etc.


`isNull()` method for a `Column` instance returns a new boolean Column:

	df_batting.withColumn('SHisNull', df_batting.SH.isNull())
	
is equivalent to `df_batting['SHisNull'] = pd.isnull(df_batting.SH)` in pandas.

The usage of `dropna()` and `fillna()` methods is the same as pandas:

	df_batting.dropna(how='any')
	df_batting.fillna({'lgID':'ValueYouLike'})
	
## Methods corresponding to ones in dplyr

### select

Use `select()` to restrict DataFrame to the specified columns

	df_career = df_batting.select(['playerID','H','AB'])

### filter

Use `filter()` to restrict DataFrame to the rows satisfying a given condition.

	df_career = df_career.filter(df_career.AB > 0)

### arrange

Use `sort()` to sort the rows of DataFrame

	df_career.sort(df_career.name, ascending=True)

### mutate 

Use `withColumn()` to create a new column. A basic usage is:

	df_career.withColumn('average', df_career.H / df_career.AB)
	
This is equivalent to `df_career['average'] = df_career.H / df_career.AB` in 
pandas **except we create a new DataFrame**. To use the new column, we have
to assign the new DataFrame to a variable, as we noticed above.

When we want to apply a function to each row, we have to create UDF. 

	from pyspark.sql.functions import udf, struct
	from pyspark.sql.types import DoubleType

	another_avg = udf(lambda row: row['H']/row['AB'], DoubleType())
	##                function applied to rows        data type of output
	df_career.withColumn("anotherAvg", 
		another_avg(struct([df_career[x] for x in df_career.columns])))

If the function taks only a few variables (columns), then we do not need
`struct()`:

	yet_another_avg = udf(lambda x,y: x/y, DoubleType())
	df_career.withColumn('yetAnotherAvg', yet_another_avg(df_career.H, df_career.AB))
		
To create a constant column we use `lit()`

	from pyspark.sql.functions import lit
	df_pitching.withColumn('Pitcher', lit(1))

This is similar to `df_pitching['Pitcher'] = 1`.


### groupby

`groupBy()` returns [GroupedData](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.GroupedData). 
`groupBy()` is equivalent to `groupby()` in pandas, but indexes form an
ordinary Column.

	df_career = df_career.groupBy('playerID').agg({'H':'sum', 'AB':'sum'})
	                     .withColumnRenamed('sum(H)','H')\
						 .withColumnRenamed('sum(AB)','AB')

**Note the column names which are generated.** The new DataFrame (with an
additional column) looks like

	df_career.withColumn('average', df_career.H / df_career.AB).show(6)
	## +---------+----+----+-------------------+
	## | playerID|  AB|   H|            average|
	## +---------+----+----+-------------------+
	## |allisdo01|1407| 381|0.27078891257995735|
	## |orourji01|8505|2643| 0.3107583774250441|
	## |gilliba01|1865| 386|0.20697050938337802|
	## |neaglja01| 369|  65|0.17615176151761516|
	## |becanbu01|  41|  10|0.24390243902439024|
	## |henglmo01| 133|  24|0.18045112781954886|
	## +---------+----+----+-------------------+
	## only showing top 6 rows


## Merge

Use [join()](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.join)
to merge two DataFrames:

	df_tmp = df_career.join(df_pitching, 'playerID', how='left')

The default value of `how` option is `inner`. 

## Pivot table

Assume that a DataFrame `df_pivot` looks like 

	## +------+------+------+
	## |yearID|teamID|sum(H)|
	## +------+------+------+
	## |  2001|   SEA|1637.0|
	## |  2009|   SLN|1436.0|
	## |  2013|   NYA|1321.0|
	## |  2008|   TOR|1453.0|
	## |  2000|   OAK|1501.0|
	## +------+------+------+
	## only showing top 5 rows

Then we can create a pivot table as follows

	df_pivot.groupBy('yearID').pivot('teamID').sum('sum(H)')
	##               index           columns   aggfunc(values)

The avove line is equivalent to the following in pandas.

	dg_pivot.pivot(index='yearID', columns='teamID', values='sum(H)', aggfunc=np.sum)

We can also restrict the columns

	needed_cols = ['ANA','TBA','WAS']
	df_pivot.groupBy('yearID').pivot('teamID', needed_cols).sum('sum(H)')
	## +------+------+------+------+
	## |yearID|   ANA|   TBA|   WAS|
	## +------+------+------+------+
	## |  2012|  null|1293.0|1468.0|
	## |  2014|  null|1361.0|1403.0|
	## |  2013|  null|1421.0|1365.0|
	#### omitted
	## |  2001|1447.0|1426.0|  null|
	## |  2010|  null|1343.0|1355.0|
	## |  2003|1473.0|1501.0|  null|
	## +------+------+------+------+

# Spark ML

[Machine Learning Library (MLlib) Guide](https://spark.apache.org/docs/latest/ml-guide.html)

- `spark.ml` : DataFrame-based APIs (primary)
- `spark.mllib` : RDD-based APIs (in maintenance)

## Terminology

- **Transformer** is a map from a DataFrame to a DataFrame by adding columns.
- **Estimator** produces a Transformer after fitting.
- **Pipeline** is a sequence of Transformers and Estimator. If it contains an
  Estimator then the Pipeline is also Estimator. Otherwise it is a Transformer.

A typical Pipeline looks like 

	DataFrame1 (id, label, anything_you_have)
	↓ Transformer1 ## create a Column of features
	DataFrame2 (id, label, features)
	↓ Transformer2 ## fitted Estimator
	DataFrame3 (id, label, features, prediction)

Note that feature variables must form *one Column* of "feature vectors":
`features` (default name). A Column of the target values is called `label`.

A typical source code looks like:
	
	from pyspark.ml import Pipeline
	from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

	transf = ... ## making a feature column, etc
	estim1 = ... ## standardize, etc
	estim2 = ... ## regression of classification estimator
	pipeline = Pipeline(stages=[transf,estim1,estim2]) ## Estimator
	
	## my_model_without_cv = pipeline.fit(df) ## training
	
	param_grid = ParamGridBuilder().addGrid(...).addGrid(...).build() # Grid

	cv_tune = CrossValidator(estimator=pipeline,
		                     estimatorParamMaps=param_grid,
                             evaluator=BinaryClassificationEvaluator(),
                             trainRatio=0.6) ## Estimator
							 
	my_model = cv_tune.fit(df) ## training. We obtain a Transformer


## Data Munging

The [pyspark.ml.feature](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#module-pyspark.ml.feature)
module contains classes for converting a DataFrame. 

Note that `pyspark.sql.*` are not a Transformer.

- **We should use functions `pyspark.sql.*` before a Pipeline.**
- If we can do a task with `pyspark.ml.*`, then we use it and avoid
  `pyspark.sql.*`. 

Remarks:

- In this document we say "*An Estimator converts ...*". This precisely means
  that "*The Transformer which is obtained by fit() converts ...*".
- An Estimator produces a Transformer by applying `fit(df)` method. The
  produced Transformer is an instance of a different class. For example an
  instance of `StandardScaler` produces an instance of
  `StandardScalerModel`. We need to care about this when we look up the
  documents.

### Bucketizer

This Transformer converts a numerical variable into a categorical variable
with respect to multiple thresholds. 

	from pyspark.ml.feature import Bucketizer, IndexToString
	splits = [-1, 5, 10, 17, 25]
	bucketizer = Bucketizer(splits=splits, inputCol='hr', outputCol='hr_idx')
	converter = IndexToString(inputCol='hr_idx', outputCol='hr_bucket')

The result of two Transformers look like 

	## +----+------+----------+
	## |  hr|hr_idx| hr_bucket|
	## +----+------+----------+
	## | 0.0|   0.0| -1.0, 5.0|
	## | 1.0|   0.0| -1.0, 5.0|
	## | 2.0|   0.0| -1.0, 5.0|
	## | 3.0|   0.0| -1.0, 5.0|
	## | 4.0|   0.0| -1.0, 5.0|
	## +----+------+----------+
	##       Double StringType

- `inputCol` must be `DoubleType`. `IntegerType` is NOT accepted. 
- If `splits = [a,b,c,d]` We have three classes: [a,b), [b,c), [c,d).
- The result is of `DoubleType`. Therefore we need to convert it into a 
  categorical variable by using `IndexToString`.
- Use `float('inf')` if the range is not upper/lower bound.

### IndexToString

This Estimator converts a numerical index to a string index. 

	from pyspark.ml.feature import StringIndexer
	indexer = StringIndexer(inputCol='label', outputCol='target')

A result of a conversion looks like:

	indexer.fit(df).transform(df).select(['index','label','target']).show()
	## +-----+---------+------+
	## |index|    label|target|
	## +-----+---------+------+
	## |  0.0|malignant|   1.0|
	## |  1.0|malignant|   1.0|
	##========================== omitted 
	## | 18.0|malignant|   1.0|
	## | 19.0|   benign|   0.0|
	## +-----+---------+------+

The Estimator `StringIndexer` recover the original Column from the converted
one. 

### VectorAssembler	

This Transformer collects several columns and make them a column

	from pyspark.ml.feature import VectorAssembler
	assembler_num = VectorAssembler(inputCols=['temp','hum','windspeed'],
	                                outputCol='num_vals')

The result looks like

	## +----+----+---------+---+---------------+
	## |temp| hum|windspeed|cnt|num_vals       |
	## +----+----+---------+---+---------------+
	## |0.24|0.81|      0.0| 16|[0.24,0.81,0.0]|
	## |0.22| 0.8|      0.0| 40|[0.22,0.8,0.0] |
	## |0.22| 0.8|      0.0| 32|[0.22,0.8,0.0] |
	## +----+----+---------+---+---------------+

### StandardScaler

This Estimator standardizes feature values. 

	from pyspark.ml.feature import StandardScaler
	scaler = StandardScaler(inputCol='num_vals', outputCol='std_num_vals')

The result looks like

    ## +---------------+-------------------------------------------+
    ## |num_vals       |std_num_vals                               |
    ## +---------------+-------------------------------------------+
    ## |[0.24,0.81,0.0]|[1.2463898755456806,4.198417543529621,0.0] |
    ## |[0.22,0.8,0.0] |[1.1425240525835405,4.146585228177403,0.0] |
    ## |[0.22,0.8,0.0] |[1.1425240525835405,4.146585228177403,0.0] |
    ## +---------------+-------------------------------------------+

### MinMaxScaler

This Estimator rescales a numerical variable in the range [0,1].
[Sample code](https://spark.apache.org/docs/latest/ml-features.html#minmaxscaler).

### RFormula

This Estimator converts feature columns and a target column in a suitable form
for machine learning. 
[Sample code](https://spark.apache.org/docs/latest/ml-features.html#rformula).
The `formula` option takes an R model formula.

	from pyspark.ml.feature import RFormula
	formula = RFormula(formula='species ~ . - index',
	                   featuresCol="features", labelCol="label")

- We do not need to create a pivot table by ourselves.
- **Check the data type** before applying `RFormula`. 

## Machine Learning

There are two modules `pyspark.ml.classification` and `pyspark.ml.regression`.
An instance of the modules is just an Estimator and its usage is very similar
to one of `sklearn`. 
[Sample codes](https://spark.apache.org/docs/latest/ml-classification-regression.html).

- The number of implemented algorithm is not large.
- A generated Transformer (aka Model) has the information about trained model.
- **Feature variables must be collected in ONE column** of vectors:
  
	    ## +-----+--------------------+-----+
	    ## |index|            features|label|
	    ## +-----+--------------------+-----+
	    ## |  0.0|[5.09999990463256...|  2.0|
	    ## |  1.0|[4.90000009536743...|  2.0|
	    ## |  2.0|[4.69999980926513...|  2.0|
	    ## |  3.0|[4.59999990463256...|  2.0|
	    ## |  4.0|[5.0,3.5999999046...|  2.0|
	    ## +-----+--------------------+-----+

- The default names of the feature column and the target column are `features`
  and `label`, respectively. To use other name, we have to explicitly specify
  them when create an Estimator instance.

### Elastic Net

	from pyspark.ml.regression import LinearRegression
	enet = LinearRegression(regParam=0.0, elasticNetParam=0.0)

The penalty term is given by

§§\lambda \left( \alpha \|w\|\_1 + (1-\alpha) \frac{\|w\|\_2^2}{2} \right) \qquad \lambda > 0,\ \alpha \in [0,1].§§

Here §\lambda§ is called an regularization parameter and §\alpha§ is called an
elastic net parameter

- The solver is "l-bfgs" (Limited-memory BFGS), "normal" (Normal Equation) or
  "auto". The default value is 'auto'. 
  ([doc](https://spark.apache.org/docs/latest/api/java/index.html))
  
  > When fitting LinearRegressionModel without intercept on dataset with constant nonzero column by “l-bfgs” solver, Spark MLlib outputs zero coefficients for constant nonzero columns.

### Penalised Logistic Regression

	from pyspark.ml.classification import LogisticRegression
	estim_plr = LogisticRegression(regParam=0.1, elasticNetParam=0.1)
	
The penalty term is the same as one of the elastic net.

### Decision Tree

	from pyspark.ml.classification import DecisionTreeClassifier
	estim_tree = DecisionTreeClassifier(maxDepth=5, seed=None)
	param_tree = ParamGridBuilder().addGrid(tree.maxDepth, [2,3,5,7,11]).build()

Regression: `from pyspark.ml.regression import DecisionTreeRegressor`

### Random Forest

	from pyspark.ml.classification import RandomForestClassifier
	estim_rf = RandomForestClassifier(maxDepth=5, numTrees=20, seed=None)

Regression: `from pyspark.ml.regression import RandomForestRegressor`


## Model Selection

We can also tune meta parameters by creating the following instances

1. a Pipeline (this describes the workflow of training)
2. a list of Param instances with ParamGridBuilder (paramter grid)
3. a CrossValidator instance (an Estimator)

### 1. Pipeline

	from pyspark.ml import Pipeline
	estim_plr = LogisticRegression(labelCol='target')
	pipline_plr = Pipeline(stages=[indexer, assembler, scaler, plr])

### 2. Param Grid

	from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

Elastic Net/Penelised Logistic Regression:

	param_plr = ParamGridBuilder().addGrid(plr.regParam, [0.01,0.1,1])\
		                          .addGrid(plr.elasticNetParam, [0,0.5,1])\
								  .build()
Decision Tree:

	param_tree = ParamGridBuilder().addGrid(tree.maxDepth, [6,12,24])\
                                   .build()

Random Forest:

	param_rf = ParamGridBuider().addGrid(rf.maxDepth, [3,5,7])\
	                            .addGrid(rf.numTrees, [10,30])\
								.build()


### 3-a. Cross-Validation

According to the problem (binary/multiclass classification or regression) we
need to specify an evaluation metric (RMSE, etc)

	from pyspark.ml.evaluation import BinaryClassificationEvaluator
	cv_plr = CrossValidator(
		estimator=pipeline_plr,        ## Pipeline
		estimatorParamMaps=param_plr,  ## paramter_grid
		evaluator=BinaryClassificationEvaluator(labelCol='target'),
		numFolds=3)
	model_plr = cv_plr.fit(df)

- We have to specify the target variable if its name is not 'label'.
- `from pyspark.ml.evaluation import BinaryClassificationEvaluator`
  - metricName : areaUnderROC (default), areaUnderPR
- `from pyspark.ml.evaluation import MulticlassClassificationEvaluator`
  - metricName : f1 (default), weightedPrecision, weightedRecall, accuracy
- `from pyspark.ml.evaluation import RegressionEvaluator`
  - metricName : rmse (default) , mse, mae 

The following function shows the performance of the trained model.

	from collections import defaultdict

	def validation_result(grid,metrics,value='score',sort=False):
		df = defaultdict(list)

		for param in grid:
			for param_obj, param_val in param.items():
				df[param_obj.name].append(param_val)

		df = pd.DataFrame(df)
		df[value] = metrics

		if sort:
			return df.sort_values(by=value, ascending=False)
		else:
			return df

	validation_result(grid_plr, model_plr.validationMetrics,sort=True)

### 3-b. Train-Validation-Split

	from pyspark.ml.tuning import TrainValidationSplit
	tvs_plr = TrainValidationSplit(estimator=pipeline_plr,
	                               estimatorParamMaps=grid_plr,
                                   evaluator=BinaryClassificationEvaluator(),
                                   trainRatio=0.7)
	model_plr = tvs_plr.fit(df)

### Clustering 

No plan to write a reference. 
[Guide](https://spark.apache.org/docs/latest/ml-clustering.html).
