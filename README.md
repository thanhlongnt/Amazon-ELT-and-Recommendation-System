## 🚀 Pipeline Summary: Amazon Review Data Processing

The pipeline transforms raw, compressed Amazon review and metadata files for various categories into a cleaned, feature-engineered dataset suitable for sequence modeling and predictive analytics.

### 1. 🛍️ Preprocess Amazon-Reviews-2023 Categories (Script 01)
This initial stage focuses on processing raw data for each Amazon category individually.

* **Input:** Raw `.jsonl.gz` review and meta files (e.g., from UCSD or Google Drive).
* **Key Operations:**
    * **Data Ingestion:** Streams and parses the gzipped JSON Lines files.
    * **User Counting:** Computes the total number of reviews/purchases for every user within that category.
    * **Exploratory Data Analysis (EDA):** Calculates basic statistics (histograms) for review ratings, helpful votes, and per-user purchase counts.
* **Outputs:**
    * `user_counts_<Category>.parquet`: Per-user purchase counts.
    * `review_stats_<Category>.json`: Summary statistics (review count, user count, verified purchases).
    * `meta_stats_<Category>.json`: Item-level statistics (item count, price mean).
    * **Visualization:** Histograms (`*_hist_<Category>.png`) for ratings, helpful votes, and user purchases.

---

### 2. 🌍 Global User Collation + User Importance Scoring (Script 02)
This stage aggregates the per-category user data to identify and score the most influential and diverse users globally.

* **Input:** `user_counts_<Category>.parquet` files from all processed categories.
* **Key Operations:**
    * **Global Aggregation:** Combines all user counts into a single user-category matrix, calculating **total purchases** and **distinct categories** per user.
    * **Diversity Scoring:** Computes **entropy** (a measure of diversity across categories) and **normalized entropy**.
    * **Importance Scoring:** Calculates a **user importance score** based on the formula:
       $$ \text{importance} = \text{total\_purchases} \times (1 + \text{norm\_entropy})$$
    * **Top User Extraction:** Filters users who meet strict criteria (e.g., importance $\ge$ 95th percentile, $\ge 3$ distinct categories, $\ge 3$ total purchases).
* **Outputs:**
    * `user_total_purchases_hist.png` and `user_distinct_categories_hist.png`.
    * `top_users.parquet`: A table of the extracted top users with their calculated importance and diversity scores.

---

### 3. 👤 Extract Top-User Filtered Features (Script 03)
This stage re-scans the raw reviews, but only retains and aggregates data for the identified **top users**.

* **Input:** `top_users.parquet` (global top users) and the raw review/meta `.jsonl.gz` files (per category).
* **Key Operations:**
    * **Review Filtering & Streaming:** Streams the raw reviews, keeping only those written by a global top user. It joins these filtered reviews with item metadata (like item average rating).
    * **Per-Review Data Generation:** Creates a detailed feature table for every filtered review.
    * **Aggregation:** Computes aggregate features for both **users** and **items** *based only on the top-user reviews*.
* **Outputs (Per Category):**
    * `top_user_reviews_<Category>.parquet`: Filtered, per-review data with item meta.
    * `top_user_features_<Category>.parquet`: Aggregated features per top user (e.g., average rating, total reviews *in this category*).
    * `top_item_features_<Category>.parquet`: Aggregated features per item (e.g., number of unique top-user reviewers).
    * **Visualization:** Rating and helpful vote histograms for the *top-user* reviews.

---

### 4. 🔗 Create Temporal Training Data (Script 04)
The final stage transforms the filtered review data into a sequential, temporal dataset for predicting the next purchase category.

* **Input:** `top_user_reviews_<Category>.parquet` and `top_user_features_<Category>.parquet` (from all successful categories).
* **Key Operations:**
    * **Sharding:** Reviews and user features are sharded by `user_id` to enable memory-efficient, parallel processing.
    * **Sequence Generation:** For each user, reviews are chronologically sorted. Training samples are created at each time step $i$ (prefix) to **predict the category of the next purchase** at time $i+1$ (label).
    * **Feature Engineering (Prefix-Based):**
        * **Static Features:** Global user importance, entropy.
        * **Sequential Features (Prefix):** Length, timespan, average rating/helpful votes, features of the *last* purchase, indices of the last $N$ categories, and counts of categories in the prefix.
    * **Baseline Computation:** Calculates simple baselines (Global Majority, Per-User Majority, Last Category) for comparison.
* **Outputs:**
    * `sequence_training_samples.parquet`: The final, global training dataset containing all engineered features and the target category index.
    * Temporary directories for sharded data (`data/tmp/`).