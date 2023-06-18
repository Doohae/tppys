from typing import Iterator

import pandas as pd
from datasets import load_dataset
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max, mean, min, pandas_udf, size
from pyspark.sql.types import (ArrayType, IntegerType, StringType, StructField,
                               StructType)

from encode import Tokenizer

SCHEMA = StructType(
    [
        StructField(name="casetype", dataType=StringType()),
        StructField(name="facts", dataType=StringType()),
    ]
)


def create_spark_session():
    spark = SparkSession.builder.master("local").appName("lbox-tokenize").getOrCreate()
    return spark


def load_data():
    column_names = ["casetype", "facts"]
    dataset = load_dataset("lbox/lbox_open", "casename_classification")["train"]
    dataset = dataset.remove_columns(
        list(set(dataset.column_names) - set(column_names))
    )
    return dataset


def main():
    spark = create_spark_session()
    ds = load_data()
    tokenizer = Tokenizer("kakaobrain/kogpt", revision="KoGPT6B-ryan1.5b-float16")
    df = spark.createDataFrame(data=ds, schema=SCHEMA)

    @pandas_udf(ArrayType(IntegerType()))
    def encode(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
        for x in batch_iter:
            yield x.apply(lambda y: tokenizer.encode(y))

    stat_df = df.withColumn("encoded", encode(col("facts"))).withColumn(
        "length", size(col("encoded"))
    )

    result = stat_df.groupBy("casetype").agg(
        mean("length").alias("mean_token_length"),
        max("length").alias("max_token_length"),
        min("length").alias("min_token_length"),
    )
    result.show()
    """
    Output must be:
    +--------+------------------+----------------+----------------+
    |casetype| mean_token_length|max_token_length|min_token_length|
    +--------+------------------+----------------+----------------+
    |   civil|     463.309765625|            4915|              31|
    |criminal|347.19485294117646|            8073|              11|
    +--------+------------------+----------------+----------------+
    """

    spark.stop()


if __name__ == "__main__":
    main()
