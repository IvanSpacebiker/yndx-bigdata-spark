package ru.yndx.bigdata.hypotheses;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.yndx.bigdata.pipeline.Datasets;
import ru.yndx.bigdata.stats.StatUtils;

import java.util.Arrays;
import java.util.List;

import static org.apache.spark.sql.functions.*;

/**
 * H10 — P(like|listen, rec) < P(like|listen, org)
 *    AND P(dislike|listen, rec) > P(dislike|listen, org).
 *    Bonferroni correction for 2 simultaneous tests.
 *
 * Also reports Asymmetry Ratio = P(dislike) / P(like) per group.
 */
public class H10Asymmetry {

    private static final Logger log = LoggerFactory.getLogger(H10Asymmetry.class);

    private final SparkSession spark;
    private final Datasets ds;

    public H10Asymmetry(SparkSession spark, Datasets ds) {
        this.spark = spark;
        this.ds    = ds;
    }

    public HypothesisResult run() {
        log.info("=== H10: Like/dislike asymmetry ===");

        // Aggregate listens by (uid, item_id)
        Dataset<Row> listenAgg = ds.listensEnriched
                .groupBy("uid", "item_id")
                .agg(last("is_organic").alias("is_organic"),
                     count("*").alias("n"));

        Dataset<Row> likeFlag = ds.likes.select("uid", "item_id").distinct()
                .withColumn("liked", lit(1));
        Dataset<Row> dislikeFlag = ds.dislikes.select("uid", "item_id").distinct()
                .withColumn("disliked", lit(1));

        Dataset<Row> h10 = listenAgg
                .join(likeFlag,    new String[] {"uid", "item_id"}, "left")
                .join(dislikeFlag, new String[] {"uid", "item_id"}, "left")
                .withColumn("liked",    coalesce(col("liked"),    lit(0)))
                .withColumn("disliked", coalesce(col("disliked"), lit(0)));

        log.info("P(like) and P(dislike) by group:");
        h10.groupBy("is_organic")
           .agg(
               count("*").alias("pairs"),
               avg("liked").alias("like_rate"),
               avg("disliked").alias("dislike_rate")
           )
           .orderBy("is_organic")
           .show();

        // Chi-squared for like
        long[][] ctLike    = buildCT(h10, "liked");
        long[][] ctDislike = buildCT(h10, "disliked");

        double[] chiLike    = StatUtils.chiSquared2x2(ctLike);
        double[] chiDislike = StatUtils.chiSquared2x2(ctDislike);

        double pLike    = chiLike[1];
        double pDislike = chiDislike[1];

        // Bonferroni: multiply p-values by 2 (2 tests)
        double pLikeAdj    = Math.min(pLike    * 2, 1.0);
        double pDislikeAdj = Math.min(pDislike * 2, 1.0);

        log.info("Chi-squared like:    χ²={}, p={}, p_adj={}", chiLike[0], pLike, pLikeAdj);
        log.info("Chi-squared dislike: χ²={}, p={}, p_adj={}", chiDislike[0], pDislike, pDislikeAdj);

        // Asymmetry ratio
        List<Row> rates = h10.groupBy("is_organic")
                .agg(avg("liked").alias("lr"), avg("disliked").alias("dr"))
                .orderBy("is_organic")
                .collectAsList();

        for (Row r : rates) {
            double lr = ((Number) r.get(1)).doubleValue(), dr = ((Number) r.get(2)).doubleValue();
            log.info("Asymmetry ratio (dislike/like) is_organic={}: {:.3f}", ((Number) r.get(0)).intValue(), lr > 0 ? dr / lr : Double.NaN);
        }

        // Directionality check from rates
        double likeRec  = ((Number) rates.get(0).get(1)).doubleValue();
        double likeOrg  = ((Number) rates.get(1).get(1)).doubleValue();
        double disRec   = ((Number) rates.get(0).get(2)).doubleValue();
        double disOrg   = ((Number) rates.get(1).get(2)).doubleValue();

        boolean likeCorrect    = likeRec    < likeOrg;
        boolean dislikeCorrect = disRec     > disOrg;
        boolean confirmed      = pLikeAdj < 0.05 && pDislikeAdj < 0.05
                              && likeCorrect && dislikeCorrect;

        String details = String.format(
                "p_like_adj=%.4e, p_dislike_adj=%.4e | like rec<org=%b, dislike rec>org=%b",
                pLikeAdj, pDislikeAdj, likeCorrect, dislikeCorrect);

        return new HypothesisResult("H10",
                "Асимметрия like/dislike для рекомендаций",
                confirmed, details);
    }

    private long[][] buildCT(Dataset<Row> df, String colName) {
        List<Row> rows = df.groupBy("is_organic", colName)
                           .count()
                           .orderBy("is_organic", colName)
                           .collectAsList();
        long[][] ct = new long[2][2];
        for (Row r : rows) {
            int org = ((Number) r.get(0)).intValue();
            int val = ((Number) r.get(1)).intValue();
            ct[org][val] = ((Number) r.get(2)).longValue();
        }
        return ct;
    }
}
