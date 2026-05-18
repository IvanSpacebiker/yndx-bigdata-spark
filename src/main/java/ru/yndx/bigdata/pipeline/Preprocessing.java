package ru.yndx.bigdata.pipeline;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;
import org.apache.spark.sql.functions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.apache.spark.sql.functions.*;

/**
 * Mirrors the Python preprocessing cells (Section 3 in the notebook):
 *
 *   3.1  reconstruct_time()  — cumsum of delta timestamps × 5 sec
 *   3.2  listen_plus flag    — played_ratio_pct ≥ 50
 *   3.3  length_bucket       — [0-2min, 2-4min, 4-7min, 7+min]
 */
public class Preprocessing {

    private static final Logger log = LoggerFactory.getLogger(Preprocessing.class);

    public static Datasets run(Datasets raw) {
        log.info("Preprocessing: reconstructing time, adding listen_plus, length_bucket...");

        // ── 3.1 Reconstruct absolute time ──────────────────────────────
        raw.listensEnriched    = reconstructTime(raw.listens);
        raw.multiEventEnriched = reconstructTime(raw.multiEvent);

        Dataset<Row> dislikesEnriched  = reconstructTime(raw.dislikes);
        Dataset<Row> likesEnriched     = reconstructTime(raw.likes);
        Dataset<Row> undislikesEn      = reconstructTime(raw.undislikes);

        // Replace originals with enriched versions (re-assign fields)
        // We keep originals for joins that don't need abs_time
        raw.listensEnriched    = addListenPlus(raw.listensEnriched);
        raw.listensEnriched    = addLengthBucket(raw.listensEnriched);

        // Store enriched dislikes/likes back via new references on Datasets
        // (Java limitation: no tuple return, so we store in listensEnriched only;
        //  hypothesis classes receive Datasets and call helpers below as needed)
        log.info("Preprocessing complete.");
        return raw;
    }

    /**
     * Reconstructs cumulative time from delta timestamps.
     *   abs_time_sec = cumsum(timestamp) * 5  per uid
     * Assumes data is sorted by (uid, timestamp) within each partition.
     */
    public static Dataset<Row> reconstructTime(Dataset<Row> df) {
        WindowSpec w = Window
                .partitionBy("uid")
                .orderBy("timestamp")
                .rowsBetween(Window.unboundedPreceding(), Window.currentRow());

        return df
                .withColumn("abs_time_sec",
                        sum("timestamp").over(w).multiply(5L));
    }

    /**
     * Adds listen_plus = (played_ratio_pct >= 50).
     * Values > 100 are valid (rewind/repeat) and count as Listen+.
     */
    public static Dataset<Row> addListenPlus(Dataset<Row> df) {
        return df.withColumn("listen_plus",
                when(col("played_ratio_pct").geq(50), 1).otherwise(0));
    }

    /**
     * Adds length_bucket column using the same bin edges as the Python notebook:
     *   [0, 120) → "0-2min"
     *   [120, 240) → "2-4min"
     *   [240, 420) → "4-7min"
     *   [420, ∞)   → "7+min"
     */
    public static Dataset<Row> addLengthBucket(Dataset<Row> df) {
        return df.withColumn("length_bucket",
                when(col("track_length_seconds").lt(120),  "0-2min")
               .when(col("track_length_seconds").lt(240),  "2-4min")
               .when(col("track_length_seconds").lt(420),  "4-7min")
               .otherwise("7+min"));
    }
}
