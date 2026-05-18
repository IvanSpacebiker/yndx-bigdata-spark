package ru.yndx.bigdata.pipeline;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

/**
 * Immutable holder for all pre-processed Spark DataFrames.
 * Fields are public-final so hypothesis classes can read them directly.
 */
public final class Datasets {

    /** Raw / unprocessed tables */
    public final Dataset<Row> listens;
    public final Dataset<Row> dislikes;
    public final Dataset<Row> likes;
    public final Dataset<Row> undislikes;
    public final Dataset<Row> unlikes;
    public final Dataset<Row> multiEvent;

    /** Enriched listens: + abs_time_sec, listen_plus, length_bucket */
    public Dataset<Row> listensEnriched;

    /** Enriched multi_event: + abs_time_sec */
    public Dataset<Row> multiEventEnriched;

    public Datasets(Dataset<Row> listens,
                    Dataset<Row> dislikes,
                    Dataset<Row> likes,
                    Dataset<Row> undislikes,
                    Dataset<Row> unlikes,
                    Dataset<Row> multiEvent) {
        this.listens    = listens;
        this.dislikes   = dislikes;
        this.likes      = likes;
        this.undislikes = undislikes;
        this.unlikes    = unlikes;
        this.multiEvent = multiEvent;
    }
}
