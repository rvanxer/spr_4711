import os, sys
from pathlib import Path

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

class Spark:
    """A custom pyspark session handler to help with pyspark operations."""
    def __init__(self, start=True):
        node = os.uname().nodename.split('.')[0]
        memory = dict(tnet1='200g', umni1='36g', umni2='36g', umni5='160g')[node]
        cores = dict(tnet1=16, umni1=20, umni2=20, umni5=32)[node]
        self.config = {
            'spark.sql.shuffle.partitions': 40,
            'spark.driver.maxResultSize': 0,
            'spark.executor.memory': '36g',
            'spark.executor.cores': 10,
            'spark.cores.max': 10,
            'spark.driver.memory': '36g',
            'spark.default.parallelism': 12,
            'spark.sql.session.timeZone': 'GMT',
            'spark.sql.debug.maxToStringFields': 100,
            'spark.sql.execution.arrow.pyspark.enabled': 'true',
            'spark.local.dir': '.tmp',
            'spark.executor.memory': memory,
            'spark.driver.memory': memory,
            'spark.default.parallelism': cores,
        }
        self.context = None
        self.session = None
        if start:
            self.start()

    def start(self):
        """Start pyspark session and store relevant objects."""
        if self.session is None:
            # set the configuration
            self.config = SparkConf().setAll(list(self.config.items()))
            # create the context
            self.context = SparkContext(conf=self.config)
            # start the session and set its log level
            self.session = SparkSession(self.context)
            self.session.sparkContext.setLogLevel('WARN')

    def read_csv(self, paths, schema=None, header=False, **kwargs):
        if isinstance(paths, str) or isinstance(paths, Path):
            paths = [paths]
        reader = self.session.read.option('header', header)
        for k, v in kwargs.items():
            reader = reader.option(k, v)
        df = reader.csv(str(paths.pop(0)), schema)
        schema_ = df.schema
        for path in paths:
            df = df.union((reader.csv(str(path), schema_)))
        return df

    def read_parquet(self, paths):
        if isinstance(paths, str) or isinstance(paths, Path):
            paths = [paths]
        return self.session.read.parquet(*[str(p) for p in paths])

    def pdf2sdf(self, df):
        df = self.session.createDataFrame(df)
        return df
