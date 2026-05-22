# Yambda-50M — Java + Gradle + Spark

Переписанный аналог ноутбука [`yndx-bigdata.ipynb`](https://github.com/ericsupersonic/yndx-bigdata).
Тестирует гипотезы H1–H11 об органических vs рекомендательных треках
в датасете [Yambda-50M](https://huggingface.co/datasets/yandex/yambda).

---

## Структура проекта

```
yndx-bigdata/
├── build.gradle
├── settings.gradle
├── README.md
└── src/main/java/ru/yndx/bigdata/
    ├── App.java                     # Точка входа
    ├── Config.java                  # Разбор аргументов CLI
    ├── pipeline/
    │   ├── DataLoader.java          # Загрузка parquet-файлов
    │   ├── Datasets.java            # Хранилище DataFrame-ов
    │   └── Preprocessing.java      # abs_time_sec, listen_plus, length_bucket
    ├── hypotheses/
    │   ├── HypothesisResult.java    # Результат одной гипотезы
    │   ├── H1ListenPlus.java        # Mann-Whitney U + rank-biserial r
    │   ├── H2DislikeRate.java       # Chi-squared + Bootstrap CI
    │   ├── H3CancellationRate.java  # Chi-squared + время отмены
    │   ├── H4Correlation.java       # Spearman ρ + Fisher z-test
    │   ├── H5LengthInteraction.java # Logistic Regression (interaction term)
    │   ├── H6RewindRate.java        # Chi-squared + sensitivity (top-1%)
    │   ├── H7SessionFatigue.java    # Сессионный анализ + Mann-Whitney
    │   ├── H8ExperiencedUsers.java  # Activity quartiles + Spearman trend
    │   ├── H9CosineSimilarity.java  # Embeddings + cosine sim → Listen+
    │   ├── H10Asymmetry.java        # Chi-squared × 2 + Bonferroni
    │   └── H11Diversity.java        # Shannon entropy + cosine diversity
    └── stats/
        └── StatUtils.java           # Mann-Whitney, Chi-sq, Bootstrap, Fisher z
```

---

## Требования

| Инструмент | Версия |
|-----------|--------|
| Java      | 11+    |
| Gradle    | 8.x (wrapper включён, или `./gradlew`) |
| Apache Spark | 3.5.x (local mode — устанавливать отдельно не нужно, входит в fat JAR) |

---

## Подготовка данных

Скачайте файлы датасета с HuggingFace в локальную папку:

```bash
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import hf_hub_download
import os

FILES = ["listens", "dislikes", "likes", "undislikes", "unlikes", "multi_event"]
os.makedirs("data", exist_ok=True)
for f in FILES:
    hf_hub_download(
        repo_id="yandex/yambda",
        filename=f"{f}.parquet",
        repo_type="dataset",
        subfolder="flat/50m",
        local_dir="data"
    )
# Опционально для H9 и H11:
hf_hub_download(
    repo_id="yandex/yambda",
    filename="embeddings.parquet",
    repo_type="dataset",
    local_dir="data"
)
EOF
```

---

## Сборка

```bash
# Сборка fat JAR
./gradlew jar

# Или стандартный distZip
./gradlew installDist
```

---

## Запуск

### Локально (local mode — для разработки/тестирования)

```bash
java -jar build/libs/yndx-bigdata-all.jar \
    --data-dir data \
    --master "local[*]"
```

### На кластере (spark-submit)

```bash
spark-submit \
    --class ru.yndx.bigdata.App \
    --master yarn \
    --deploy-mode cluster \
    --num-executors 20 \
    --executor-memory 8g \
    --executor-cores 4 \
    build/libs/yndx-bigdata-all.jar \
    --data-dir hdfs:///user/data/yambda-50m \
    --master yarn
```

### На Hadoop (HDFS)

```bash
# Загрузить данные в HDFS
hdfs dfs -mkdir -p /user/data/yambda-50m
hdfs dfs -put data/*.parquet /user/data/yambda-50m/

# Запустить
spark-submit \
    --class ru.yndx.bigdata.App \
    build/libs/yndx-bigdata-all.jar \
    --data-dir hdfs:///user/data/yambda-50m
```

---

## Дополнительные параметры

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--data-dir` | `data` | Путь к папке с parquet-файлами (локальный или HDFS) |
| `--master` | `local[*]` | Spark master URL |
| `--seed` | `42` | Случайный seed |
| `--sample-size` | `500000` | Размер выборки для дорогих тестов (Mann-Whitney) |

---

## Соответствие Python → Java

| Ноутбук (Python)                  | Java-класс                     |
|-----------------------------------|-------------------------------|
| `reconstruct_time()`             | `Preprocessing.reconstructTime()` |
| `listen_plus` флаг               | `Preprocessing.addListenPlus()` |
| `length_bucket` бакеты           | `Preprocessing.addLengthBucket()` |
| `mannwhitneyu` (scipy)           | `StatUtils.mannWhitneyLess/TwoSided()` |
| `chi2_contingency` (scipy)       | `StatUtils.chiSquared2x2()` |
| Bootstrap CI (numpy)             | `StatUtils.bootstrapDiffCI()` |
| `spearmanr` (scipy)              | `StatUtils.spearman()` |
| `fisher_z_test` (custom)         | `StatUtils.fisherZTest()` |
| `LogisticRegression` (sklearn)   | `spark.ml.LogisticRegression` |

---

## Гипотезы

| ID  | Гипотеза | Метод |
|-----|----------|-------|
| H1  | Listen+ ниже для рекомендаций | Mann-Whitney U + rank-biserial r |
| H2  | P(dislike) выше для рекомендаций | Chi-squared + Bootstrap CI |
| H3  | Дизлайки рекомендаций чаще отменяются | Chi-squared + время до отмены |
| H4  | Корреляция ratio–like слабее для рекомендаций | Spearman ρ + Fisher z-test |
| H5  | Listen+ падает быстрее с длиной для рекомендаций | Logistic Regression (interaction) |
| H6  | Перемотка (ratio>100%) чаще у органики | Chi-squared + sensitivity top-1% |
| H7  | P(dislike) растёт к концу рек-сессий | Сессионный анализ + Mann-Whitney |
| H8  | У опытных пользователей рекомендации эффективнее | Quartile analysis + Spearman trend |
| H9  | Похожие на органику рекомендации лучше дослушиваются | Cosine similarity + Spearman ρ |
| H10 | Асимметрия: меньше лайков, больше дизлайков | Chi-squared × 2 + Bonferroni |
| H11 | Рекомендации увеличивают разнообразие | Shannon entropy + cosine diversity |
