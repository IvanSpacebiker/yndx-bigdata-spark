package ru.yndx.bigdata.hypotheses;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.yndx.bigdata.pipeline.Datasets;
import ru.yndx.bigdata.stats.StatUtils;

import java.util.List;

import static org.apache.spark.sql.functions.*;

/**
 * H4 — Spearman ρ(played_ratio_pct, liked) is significantly weaker
 *       for recommended tracks than for organic tracks.
 *
 * Method:
 *   1. Aggregate listens by (uid, item_id): max ratio, is_organic (last)
 *   2. Left-join likes → liked flag
 *   3. Spearman ρ per group (sample ≤ 300k)
 *   4. Fisher z-test to compare the two correlations
 */
public class H4Correlation {

    private static final Logger log = LoggerFactory.getLogger(H4Correlation.class);
    private static final int SAMPLE_SIZE = 100_000;
    private static final long SEED = 42L;

    private final SparkSession spark;
    private final Datasets ds;

    public H4Correlation(SparkSession spark, Datasets ds) {
        this.spark = spark;
        this.ds    = ds;
    }

    public HypothesisResult run() {
        log.info("=== H4: Spearman rho(played_ratio, like) ===");

        // 1. Aggregate
        Dataset<Row> listenAgg = ds.listensEnriched
                .groupBy("uid", "item_id")
                .agg(
                    max("played_ratio_pct").alias("played_ratio"),
                    last("is_organic").alias("is_organic")
                );

        // 2. Like flag
        Dataset<Row> likeFlag = ds.likes
                .select("uid", "item_id").distinct()
                .withColumn("liked", lit(1));

        Dataset<Row> h4 = listenAgg
                .join(likeFlag, new String[] {"uid", "item_id"}, "left")
                .withColumn("liked", coalesce(col("liked"), lit(0)))
                .filter(col("played_ratio").isNotNull());

        // 3. Spearman ρ per group (collect sample)
        double[] rhoRec = spearmanForGroup(h4, 0);
        double[] rhoOrg = spearmanForGroup(h4, 1);

        log.info("Spearman rho: rec={} (p={}), org={} (p={})",
                rhoRec[0], rhoRec[1], rhoOrg[0], rhoOrg[1]);

        // 4. Fisher z-test
        int nRec = getSampleCount(h4, 0);
        int nOrg = getSampleCount(h4, 1);
        double[] fisher = StatUtils.fisherZTest(rhoRec[0], rhoOrg[0], nRec, nOrg);

        log.info("Fisher z-test: z={}, p={}", fisher[0], fisher[1]);

        boolean confirmed = fisher[1] < 0.05 && rhoRec[0] < rhoOrg[0];
        String details = String.format(
                "ρ_rec=%.4f, ρ_org=%.4f | Fisher z=%.3f, p=%.4e",
                rhoRec[0], rhoOrg[0], fisher[0], fisher[1]);

        return new HypothesisResult("H4",
                "Корреляция ratio–like слабее для рекомендаций",
                confirmed, details);
    }

    private double[] spearmanForGroup(Dataset<Row> df, int isOrganic) {
        List<Row> rows = df.filter(col("is_organic").equalTo(isOrganic))
                           .select("played_ratio", "liked")
                           .sample(false, 0.5, SEED)
                           .limit(SAMPLE_SIZE)
                           .collectAsList();

        double[] ratios = new double[rows.size()];
        double[] liked  = new double[rows.size()];
        for (int i = 0; i < rows.size(); i++) {
            ratios[i] = ((Number) rows.get(i).get(0)).doubleValue();
            liked[i]  = ((Number) rows.get(i).get(1)).intValue();
        }
        return StatUtils.spearman(ratios, liked);
    }

    private int getSampleCount(Dataset<Row> df, int isOrganic) {
        return (int) Math.min(SAMPLE_SIZE,
                df.filter(col("is_organic").equalTo(isOrganic)).count());
    }
}
