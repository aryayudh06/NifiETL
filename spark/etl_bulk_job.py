import argparse
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.sql.types import IntegerType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_mongo_output(output_uri):
    """
    Parse output URI in the form:
    mongodb://host:port/database.collection
    Returns (mongo_uri, database, collection)
    """
    if not output_uri.startswith("mongodb://"):
        raise ValueError("Output must start with mongodb://")

    # Pisahkan host dari path database.collection
    try:
        base_uri, path = output_uri.rsplit("/", 1)
        db_name, collection_name = path.split(".", 1)
    except ValueError:
        raise ValueError("Output must be in format: mongodb://host:port/database.collection")

    mongo_uri = f"{base_uri}/{db_name}"
    return mongo_uri, db_name, collection_name


def main(input_path, output_uri):
    mongo_uri, database, collection = parse_mongo_output(output_uri)
    logger.info(f"Parsed Mongo URI: {mongo_uri}")
    logger.info(f"Target Database: {database}")
    logger.info(f"Target Collection: {collection}")

    spark = SparkSession.builder \
        .appName("NYC Taxi ETL") \
        .config("spark.mongodb.write.connection.uri", mongo_uri) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()

    # Read CSV files
    df_fhv = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{input_path}/*/fhv_tripdata_2025-*.csv")
    df_green = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{input_path}/*/green_tripdata_2025-*.csv")
    df_yellow = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{input_path}/*/yellow_tripdata_2025-*.csv")
    df_fhvhv = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{input_path}/*/fhvhv_tripdata_2025-*.csv")

    # Log counts
    logger.info(f"Initial counts - FHV: {df_fhv.count()}, Green: {df_green.count()}, Yellow: {df_yellow.count()}, FHVHV: {df_fhvhv.count()}")

    # Add trip_type column
    df_yellow = df_yellow.withColumn("trip_type", col("RatecodeID").cast(IntegerType()))
    df_green = df_green.withColumn("trip_type", col("trip_type").cast(IntegerType()))
    df_fhv = df_fhv.withColumn("trip_type", col("SR_Flag").cast(IntegerType()))
    df_fhvhv = df_fhvhv.withColumn("trip_type", lit("FHVHV"))

    # Select and rename columns
    df_yellow = df_yellow.selectExpr(
        "tpep_pickup_datetime as pickup_datetime",
        "tpep_dropoff_datetime as dropoff_datetime",
        "PULocationID", "DOLocationID", "trip_type"
    )
    df_green = df_green.selectExpr(
        "lpep_pickup_datetime as pickup_datetime",
        "lpep_dropoff_datetime as dropoff_datetime",
        "PULocationID", "DOLocationID", "trip_type"
    )
    df_fhv = df_fhv.selectExpr(
        "pickup_datetime",
        "dropOff_datetime as dropoff_datetime",
        "PUlocationID as PULocationID",
        "DOlocationID as DOLocationID",
        "trip_type"
    )
    df_fhvhv = df_fhvhv.selectExpr(
        "pickup_datetime", "dropoff_datetime", "PULocationID", "DOLocationID", "trip_type"
    )

    # Ensure consistent data types
    for df in [df_yellow, df_green, df_fhv, df_fhvhv]:
        df = df.withColumn("PULocationID", col("PULocationID").cast(IntegerType()))
        df = df.withColumn("DOLocationID", col("DOLocationID").cast(IntegerType()))

    # Union all
    unified_df = df_yellow.unionByName(df_green, allowMissingColumns=True) \
                          .unionByName(df_fhv, allowMissingColumns=True) \
                          .unionByName(df_fhvhv, allowMissingColumns=True)

    logger.info("Sample of unified data:")
    unified_df.show(5)
    logger.info(f"Unified DataFrame count: {unified_df.count()}")

    # Write to MongoDB
    logger.info(f"Writing to MongoDB collection: {collection}")
    unified_df.write \
        .format("mongodb") \
        .mode("append") \
        .option("database", database) \
        .option("collection", collection) \
        .save()

    logger.info("ETL job completed successfully!")
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input folder path containing CSV files")
    parser.add_argument("--output", required=True, help="MongoDB URI in format mongodb://host:port/database.collection")
    args = parser.parse_args()

    try:
        main(args.input, args.output)
    except Exception as e:
        logger.error(f"ETL failed: {e}", exc_info=True)
        raise
