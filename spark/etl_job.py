import argparse
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.sql.types import IntegerType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(input_path, mongo_uri):
    spark = SparkSession.builder \
        .appName("NYC Taxi ETL") \
        .config("spark.mongodb.write.connection.uri", mongo_uri) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()

    # Read CSV files with correct schema inference
    df_fhv = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{input_path}/fhv_tripdata_2025-01.csv")
    df_green = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{input_path}/green_tripdata_2025-01.csv")
    df_yellow = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{input_path}/yellow_tripdata_2025-01.csv")
    df_fhvhv = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{input_path}/fhvhv_tripdata_2025-01.csv")

    # Log initial counts
    logger.info(f"Initial counts - FHV: {df_fhv.count()}, Green: {df_green.count()}, Yellow: {df_yellow.count()}, FHVHV: {df_fhvhv.count()}")

    # Add trip_type column for each dataset
    df_yellow = df_yellow.withColumn("trip_type", col("RatecodeID").cast(IntegerType()))
    df_green = df_green.withColumn("trip_type", col("trip_type").cast(IntegerType()))
    df_fhv = df_fhv.withColumn("trip_type", col("SR_Flag").cast(IntegerType()))
    df_fhvhv = df_fhvhv.withColumn("trip_type", lit("FHVHV"))  # Using license as identifier

    # Select and rename columns with CORRECT column names from your CSV
    df_yellow = df_yellow.selectExpr(
        "tpep_pickup_datetime as pickup_datetime",
        "tpep_dropoff_datetime as dropoff_datetime",
        "PULocationID", 
        "DOLocationID", 
        "trip_type"
    )
    
    df_green = df_green.selectExpr(
        "lpep_pickup_datetime as pickup_datetime",
        "lpep_dropoff_datetime as dropoff_datetime", 
        "PULocationID", 
        "DOLocationID", 
        "trip_type"
    )
    
    # FIXED: Use correct column names from your CSV
    df_fhv = df_fhv.selectExpr(
        "pickup_datetime",
        "dropOff_datetime as dropoff_datetime",  # Match CSV column name
        "PUlocationID as PULocationID",          # Match CSV column name  
        "DOlocationID as DOLocationID",          # Match CSV column name
        "trip_type"
    )
    
    df_fhvhv = df_fhvhv.selectExpr(
        "pickup_datetime", 
        "dropoff_datetime",
        "PULocationID", 
        "DOLocationID", 
        "trip_type"
    )

    # Ensure consistent data types
    for df in [df_yellow, df_green, df_fhv, df_fhvhv]:
        df = df.withColumn("PULocationID", col("PULocationID").cast(IntegerType()))
        df = df.withColumn("DOLocationID", col("DOLocationID").cast(IntegerType()))

    # Union all datasets
    unified_df = df_yellow.unionByName(df_green, allowMissingColumns=True) \
                          .unionByName(df_fhv, allowMissingColumns=True) \
                          .unionByName(df_fhvhv, allowMissingColumns=True)

    # Log sample data
    logger.info("Sample of unified data:")
    unified_df.show(5)
    logger.info(f"Unified DataFrame count: {unified_df.count()}")

    # Write to MongoDB
    logger.info("Writing to MongoDB...")
    unified_df.write \
        .format("mongodb") \
        .mode("append") \
        .option("database", "nyc_taxi") \
        .option("collection", "trips_2025_01") \
        .save()

    logger.info("ETL job completed successfully!")
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    try:
        main(args.input, args.output)
    except Exception as e:
        logger.error(f"ETL failed: {e}", exc_info=True)
        raise