package ru.yndx.bigdata.pipeline;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Paths;

/**
 * Loads Yambda-50M parquet files into Spark DataFrames.
 *
 * Expected schema of listens.parquet:
 *   uid LONG, item_id LONG, timestamp LONG,
 *   is_organic INT, played_ratio_pct DOUBLE,
 *   track_length_seconds DOUBLE
 *
 * Expected schema of multi_event.parquet:
 *   uid LONG, item_id LONG, timestamp LONG,
 *   is_organic INT, event_type STRING
 */
public class DataLoader {

    private static final Logger log = LoggerFactory.getLogger(DataLoader.class);

    private final SparkSession spark;
    private final String dataDir;

    public DataLoader(SparkSession spark, String dataDir) {
        this.spark   = spark;
        this.dataDir = dataDir;
    }

    public Datasets load() {
        log.info("Loading datasets from {}", dataDir);

        Dataset<Row> listens    = read("listens.parquet");
        Dataset<Row> dislikes   = read("dislikes.parquet");
        Dataset<Row> likes      = read("likes.parquet");
        Dataset<Row> undislikes = read("undislikes.parquet");
        Dataset<Row> unlikes    = read("unlikes.parquet");
        Dataset<Row> multiEvent = read("multi_event.parquet");

        logCount("listens",    listens);
        logCount("dislikes",   dislikes);
        logCount("likes",      likes);
        logCount("undislikes", undislikes);
        logCount("multi_event",multiEvent);

        return new Datasets(listens, dislikes, likes, undislikes, unlikes, multiEvent);
    }

    private Dataset<Row> read(String filename) {
        String path = Paths.get(dataDir, filename).toString();
        log.info("  Reading {}", path);
        return spark.read().parquet(path);
    }

    private void logCount(String name, Dataset<Row> ds) {
        log.info("  {} → {} rows, {} unique users",
                name, ds.count(), ds.select("uid").distinct().count());
    }
}
