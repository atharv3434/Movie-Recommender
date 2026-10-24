from pyspark.sql import SparkSession


def get_spark(app_name: str = "recommender") -> SparkSession:
    spark = (SparkSession.builder.appName(app_name).master("local[*]")
             .config("spark.sql.shuffle.partitions", "8")
             .config("spark.driver.memory", "3g")
             .config("spark.ui.showConsoleProgress", "false")
             .config("spark.sql.crossJoin.enabled", "true")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark
