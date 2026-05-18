package ru.yndx.bigdata.hypotheses;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.yndx.bigdata.pipeline.Datasets;
import ru.yndx.bigdata.stats.StatUtils;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.apache.spark.sql.functions.*;

/**
 * H6 — P(played_ratio_pct > 100 | is_organic=1) > P(... | is_organic=0).
 *       Organic listeners replay/rewind more.
 *
 * Includes sensitivity check: exclude top-1% heavy rewind users.
 */
public class H6RewindRate {

    private static final Logger log = LoggerFactory.getLogger(H6RewindRate.class);

    private final SparkSession spark;
    private final Datasets ds;

    public H6RewindRate(SparkSession spark, Datasets ds) {
        this.spark = spark;
        this.ds    = ds;
    }

    public HypothesisResult run() {
        log.info("=== H6: P(played_ratio > 100%) ===");

        Dataset<Row> listens = ds.listensEnriched
                .withColumn("rewind",
                        when(col("played_ratio_pct").gt(100), 1).otherwise(0));

        log.info("P(rewind) by group:");
        listens.groupBy("is_organic")
               .agg(
                   count("*").alias("total"),
                   sum("rewind").alias("rewind_count"),
                   avg("rewind").alias("rewind_rate")
               )
               .orderBy("is_organic")
               .show();

        // Chi-squared
        long[][] ct = buildCT(listens);
        double[] chi = StatUtils.chiSquared2x2(ct);
        double chi2 = chi[0], pChi2 = chi[1], or = chi[2];

        log.info("Chi-squared: χ²={}, p={}, OR(org/rec)={}", chi2, pChi2, or);

        // Concentration check: top-1% users
        Dataset<Row> rewindUsers = listens.filter(col("rewind").equalTo(1))
                .groupBy("uid").count().orderBy(desc("count"));

        long totalRewindUsers = rewindUsers.count();
        int top1Count = (int)(totalRewindUsers * 0.01) + 1;

        List<Row> topRows = rewindUsers.limit(top1Count).select("uid").collectAsList();
        Set<Long> top1Uids = new HashSet<>();
        for (Row r : topRows) top1Uids.add(((Number) r.get(0)).longValue());

        long totalRewindEvents = listens.filter(col("rewind").equalTo(1)).count();
        // (approximate top-1% share — only need it for logging)
        log.info("Sensitivity: top-1% users ({} users)", top1Count);

        log.info("P(rewind) excluding top-1% users:");
        // Use Spark broadcast for the filter
        Dataset<Row> listensNoTop = listens.filter(
                not(col("uid").isin(top1Uids.toArray()))
        );
        listensNoTop.groupBy("is_organic")
                    .agg(avg("rewind").alias("rewind_rate"))
                    .orderBy("is_organic")
                    .show();

        boolean confirmed = pChi2 < 0.05 && or > 1.0;   // OR > 1 → organic rewinde more
        String details = String.format(
                "χ²=%.2f, p=%.4e, OR(org/rec)=%.4f", chi2, pChi2, or);

        return new HypothesisResult("H6",
                "Перемотка (ratio>100%) чаще у органики",
                confirmed, details);
    }

    private long[][] buildCT(Dataset<Row> df) {
        List<Row> rows = df.groupBy("is_organic", "rewind")
                           .count()
                           .orderBy("is_organic", "rewind")
                           .collectAsList();
        long[][] ct = new long[2][2];
        for (Row r : rows) {
            int org    = ((Number) r.get(0)).intValue();
            int rewind = ((Number) r.get(1)).intValue();
            ct[org][rewind] = ((Number) r.get(2)).longValue();
        }
        return ct;
    }
}
