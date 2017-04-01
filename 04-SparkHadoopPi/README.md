# Apache Spark on Hadoop on Raspberry Pi

- This is just notes on installation, settings and test scripts for working on
  Apache Spark.
- The Raspberry Pi is a standalone (single-node cluster)
- `word_counting.ipynb` shows a few examples of MapReduce procedures.

## Raspberry Pi

- Raspberry Pi 3 - Model B
  ([Vilros Raspberry Pi 3 Ultimate Starter Kit--Clear Case Edition](http://www.vilros.com/vilros-raspberry-pi-3-ultimate-starter-kit-clear-case-32gb-sd-card-edition.html))
- A USB stick (64GB) is used as a storage.
- Headless (i.e. without a monitor)
- OS: [Raspbian January 2017](https://www.raspberrypi.org/downloads/raspbian/)
- [How to enable an SSH server](https://www.raspberrypi.org/documentation/remote-access/ssh/): 
  > For headless setup, SSH can be enabled by placing a file named 'ssh', without any extension, onto the boot partition of the SD card.

  It is OK to put an empty file. 
- [Raspberry Pi Foundation](https://www.raspberrypi.org/)
- [Raspberry Pi Stack Exchenge](http://raspberrypi.stackexchange.com/)

It is better to enlarge the swap size. (This is needed to install an R package.)

1. In `/etc/dphys-swapfile`:

		CONF_SWAPSIZE=1000
2. `$ sudo /etc/init.d/dphys-swapfile restart`

## Jupyter 

- [Running a notebook server](http://jupyter-notebook.readthedocs.io/en/latest/public_server.html)
- We can use a terminal on Jupyter.
- `xgboost` [is NOT avilable because of the processor](https://github.com/dmlc/xgboost/issues/1921).

## Hadoop 

- [Official Document](http://hadoop.apache.org/docs/stable/),
  [Command Guide](http://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/CommandsManual.html)
- [Hadoop Tutorial](https://www.tutorialspoint.com/hadoop/index.htm),
  [Command Reference](https://www.tutorialspoint.com/hadoop/hadoop_command_reference.htm)
- [Install Hadoop 2 on Raspberry Pi](https://vankoo.wordpress.com/2015/05/20/install-hadoop-2-raspberry/), [[ISSUE] Hadoop start-dfs Error On 32-bit Virtual Machine [FIXED]](https://mfaizmzaki.com/2015/08/26/issue-hadoop-start-dfs-error-on-32-bit-virtual-machine-fixed/)
- [A Guide to Python Frameworks for Hadoop](http://blog.cloudera.com/blog/2013/01/a-guide-to-python-frameworks-for-hadoop/)

### Hadoop Streaming

We may use any executable scripts: `mapper` and `reducer`. Both scripts
receive a file path and return key-value pairs to the standard output.

	$ hadoop jar /opt/hadoop-2.7.3/share/hadoop/tools/lib/hadoop-streaming-2.7.3.jar
	> -mapper mapper.pl -reducer reducer.pl \
	> -input word_count -output word_count_output \
	> -file mapper.pl -file reducer.pl

- [Hadoop Streaming](http://hadoop.apache.org/docs/stable/hadoop-streaming/HadoopStreaming.html)
- To use Hadoop from other user, set the right permission to the temporary
  directory in HDFS. [Hint(?)](http://stackoverflow.com/a/23601455/2205667)
- [Writing an Hadoop MapReduce Program in Python](http://www.michael-noll.com/tutorials/writing-an-hadoop-mapreduce-program-in-python/)
- **A script in [Python3 is not working](http://serverfault.com/q/807839)**.


### mrjob

- [mrjob](https://pythonhosted.org/mrjob/) 
  ([GitHub](https://github.com/Yelp/mrjob))
- We do not need to store input files on HDFS in advance.


## Apache Spark
  
- [Official Document](http://spark.apache.org/docs/latest/),
- [Setting up a standalone Apache Spark cluster of Raspberry Pi 2](https://darrenjw2.wordpress.com/2015/04/18/setting-up-a-standalone-apache-spark-cluster-of-raspberry-pi-2/)
- [Apache Spark Tutorial](https://www.tutorialspoint.com/apache_spark/).
  Relatively old
- [PySpark: How to install and Integrate with the Jupyter Notebook](https://www.dataquest.io/blog/pyspark-installation-guide/)

### PySpark

- [PySpark API Docs](http://spark.apache.org/docs/latest/api/python/index.html)
- MLlib
  [Guide](http://spark.apache.org/docs/latest/ml-guide.html),
  [Documentation](http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html)

To use PySpark-API, especially MLlib, from Python3, we need to modify some
Python scripts in the Spark tar ball. In the directory 
`spark-2.1.0-bin-hadoop2.7/examples/src/main/python/{streaming|mllib}` there
are several scripts which are not valid in Python3 and these cause errors
when installing PySpark. So we need to fix them. The invalid lines are all
of form 

	lambda (v, p): ......... ## valid only for Python 2

This must be 

	lambda v, p: ......

`spark-for-py3.txt` is a patch to fix such lamda statements.

To use Python3 (instead of Python2) for the PySpark-shell we need the
following environment.

	export PYSPARK_PYTHON=python3

In the PySpark-shell a `SparkContext` instance is avilable as a variable `sc`.

We can also use the PySpark on Jupyter 

	bash$ PYSPARK_DRIVER_PYTHON="jupyter-notebook" pyspark

We can also use PySpark not through PySpark-shell by importing `pyspark.SparkContext`:

	from pyspark import SparkContext
	sc = SparkContext() # this takes a while


