package ru.yndx.bigdata;

import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.yndx.bigdata.hypotheses.*;
import ru.yndx.bigdata.pipeline.DataLoader;
import ru.yndx.bigdata.pipeline.Datasets;
import ru.yndx.bigdata.pipeline.Preprocessing;

/**
 * Entry point. Run via spark-submit or locally.
 *
 * Usage:
 *   spark-submit --class ru.yndx.bigdata.App yndx-bigdata-all.jar \
 *       --data-dir /path/to/yambda-50m \
 *       --hypotheses all
 *
 * --data-dir must contain:
 *   listens.parquet, dislikes.parquet, likes.parquet,
 *   undislikes.parquet, unlikes.parquet, multi_event.parquet
 *   (optionally: embeddings.parquet for H9, H11)
 */
public class App {

    private static final Logger log = LoggerFactory.getLogger(App.class);

    public static void main(String[] args) {
        Config cfg = Config.parse(args);

        SparkSession spark = SparkSession.builder()
                .appName("Yambda-50M Hypothesis Testing")
                .config("spark.sql.shuffle.partitions", "200")
                .config("spark.sql.parquet.enableVectorizedReader", "true")
                // Run locally when no cluster is configured
                .master(cfg.master)
                .getOrCreate();

        spark.sparkContext().setLogLevel("WARN");

        log.info("=== Yambda-50M: Starting hypothesis pipeline ===");
        log.info("Data dir: {}", cfg.dataDir);

        // ── 1. Load ──────────────────────────────────────────────────
        DataLoader loader = new DataLoader(spark, cfg.dataDir);
        Datasets raw = loader.load();

        // ── 2. Preprocess ────────────────────────────────────────────
        Datasets ds = Preprocessing.run(raw);

        // ── 3. Hypotheses ─────────────────────────────────────────────
        HypothesisResult[] results = new HypothesisResult[]{
                new H1ListenPlus(spark, ds).run(),
                new H2DislikeRate(spark, ds).run(),
                new H3CancellationRate(spark, ds).run(),
                new H4Correlation(spark, ds).run(),
                new H5LengthInteraction(spark, ds).run(),
                new H6RewindRate(spark, ds).run(),
                new H7SessionFatigue(spark, ds).run(),
                new H8ExperiencedUsers(spark, ds).run(),
                new H9CosineSimilarity(spark, ds, cfg.dataDir).run(),
                new H10Asymmetry(spark, ds).run(),
                new H11Diversity(spark, ds).run(),
        };

        // ── 4. Summary ────────────────────────────────────────────────
        printSummary(results);

        spark.stop();
    }

    private static void printSummary(HypothesisResult[] results) {
        System.out.println("\n" + "=".repeat(65));
        System.out.println("ИТОГОВАЯ СВОДКА");
        System.out.println("=".repeat(65));
        for (HypothesisResult r : results) {
            String mark = r.confirmed ? "✅ ПОДТВЕРЖДЕНА" : "❌ НЕ ПОДТВЕРЖДЕНА";
            System.out.printf("%n%s — %s:%n  %s%n  %s%n",
                    r.id, r.description, mark, r.details);
        }
        System.out.println("\n" + "=".repeat(65));
        long confirmed = java.util.Arrays.stream(results).filter(r -> r.confirmed).count();
        System.out.printf("Подтверждено: %d / %d%n", confirmed, results.length);
        System.out.println("=".repeat(65));
    }
}
