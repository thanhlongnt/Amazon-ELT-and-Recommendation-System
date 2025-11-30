"""
1. downlaod all categories
2. concat them all together
3. create user_id -> purchase_history (list of purchase ids sorted by time)
4. create user_id -> purchase_history_categories (list of purchased categories sorted by time)
5. get the top n most recent and one hot encode them
6. feature vector 
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

import data_io
from tqdm import tqdm
import os

class DataLoader:
    def ensure_categories_downloaded(self, categories:List[str]):
        succuessful_categories = []
        for cat in categories:
            try:
                data_io.ensure_local_path(f"data/processed/{cat}/top_item_features_{cat}.parquet")
                data_io.ensure_local_path(f"data/processed/{cat}/top_user_features_{cat}.parquet")
                data_io.ensure_local_path(f"data/processed/{cat}/top_user_reviews_{cat}.parquet")
                succuessful_categories.append(cat)
            except:
                print(f"top user {cat} not processed yet")
                with open('data/processed/missing_categories.txt', 'a') as f:
                    f.write(f"{cat}\n")

        return succuessful_categories


    def load_categories_from_file(self, file_path: str) -> List[str]:
        """Load categories from a line-separated text file."""
        with open(file_path, 'r') as f:
            categories = [line.strip() for line in f if line.strip()]
        return categories
    
    def build_one_hot(self, categories):
        one_hot_map = {}
        
        # Unknown should be 0
        i = 1
        for cat in categories:
            if cat != 'Unknown':
                one_hot_map[cat] = i
                i += 1

        return one_hot_map


class DataPipeLine():
    def get_all_top_item_features(self, categories):
        all_dfs = []

        for cat in categories:
            try:
                df = pd.read_parquet(f"data/processed/{cat}/top_item_features_{cat}.parquet")
                df['category'] = cat

                print(f"Loaded category: {cat}, shape: {df.shape}")
                all_dfs.append(df)
            except FileNotFoundError:
                print(f"Warning: Could not find data for category {cat}")
                continue
        
        if not all_dfs:
            print("No data found for any categories")
            return {}
        
        combined_df = pd.concat(all_dfs, ignore_index=True)

        return combined_df


    def create_user_history(self, categories: List[str]) -> Dict[str, List[str]]:
        all_dfs = []

        for cat in categories:
            try:
                df = pd.read_parquet(f"data/processed/{cat}/top_user_reviews_{cat}.parquet")
                print(f"Loaded category: {cat}, shape: {df.shape}")
                all_dfs.append(df)
            except FileNotFoundError:
                print(f"Warning: Could not find data for category {cat}")
                continue
        
        if not all_dfs:
            print("No data found for any categories")
            return {}
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Combined data shape: {combined_df.shape}")
        
        combined_df_sorted = combined_df.sort_values(['user_id', 'unixReviewTime'])
        
        user_purchase_map = combined_df_sorted.groupby('user_id')['product_id'].apply(list).to_dict()
        
        print(f"Total unique users across all categories: {len(user_purchase_map)}")
        
        if user_purchase_map:
            sample_user = list(user_purchase_map.keys())[0]
            print(f"Sample user {sample_user} purchased: {user_purchase_map[sample_user][:5]}...")
            print("-" * 50)
        
        return user_purchase_map
    
    def build_user_characteristics(self, categories):
        all_dfs = []

        for cat in categories:
            try:
                df = pd.read_parquet(f"data/processed/{cat}/top_user_features_{cat}.parquet")
                print(f"Loaded category: {cat}, shape: {df.shape}")
                all_dfs.append(df)
            except FileNotFoundError:
                print(f"Warning: Could not find data for category {cat}")
                continue
        
        if not all_dfs:
            print("No data found for any categories")
            return {}
        
        combined_df = pd.concat(all_dfs, ignore_index=True)

        combined_df = combined_df.drop_duplicates(subset=['user_id'], keep='first')
        print(f"Combined data shape after removing duplicates: {combined_df.shape}")

        # Select only the columns we need
        columns_to_extract = [
            'user_id',
            'avg_item_avg_rating',
            'first_review_time',
            'last_review_time',
            'total_purchases',
            'distinct_categories',
            'entropy',
            'norm_entropy',
            'importance'
        ]
        
        # Filter to only existing columns
        df_filtered_columns = [col for col in columns_to_extract if col in combined_df.columns]
        missing_columns = [col for col in columns_to_extract if col not in combined_df.columns]
        
        if missing_columns:
            print(f"Warning: Columns not found: {missing_columns}")
        
        result_df = combined_df[df_filtered_columns]
        
        return result_df

def create_training_samples(feature_df, user_purchase_map, df_item_features, category_one_hot_map):

    product_to_category = df_item_features.set_index('product_id')['category'].to_dict()

    all_columns = list(feature_df.columns)
    user_id_idx = all_columns.index("user_id")

    rows = []

    for row in tqdm(feature_df.itertuples(index=False), total=len(feature_df), desc="Creating training samples"):
        user_id = row[user_id_idx]
        purchase_history = user_purchase_map.get(user_id, [])

        n = len(purchase_history)
        if n <= 1: 
            continue

        static_features = list(row)

        for i in range(1, n):
            target_product = purchase_history[i]
            target_category = product_to_category.get(target_product, "Unknown")
            target_label = category_one_hot_map.get(target_category, 0)

            rows.append(static_features + [target_label, i])

    training_df = pd.DataFrame(
        rows,
        columns=all_columns + ["target_category", "sample_index"]
    )

    # Build feature/target splits
    feature_columns = [c for c in training_df.columns 
                       if c not in ("user_id", "target_category", "sample_index")]

    X = training_df[feature_columns]
    y = training_df["target_category"]

    print(f"Created {len(training_df)} training samples")
    print(f"Feature matrix shape: {X.shape}")
    print("Target distribution:")
    print(y.value_counts())

    return X, y, training_df

def build_top_n_features(category_one_hot_map, user_purchase_map, df_user_chars, df_item_features, n_latest=10):
    """Optimized version using vectorized operations."""
    
    product_to_category = df_item_features.set_index('product_id')['category'].to_dict()
    category_to_idx = {cat: idx for idx, cat in enumerate(category_one_hot_map.keys(), 1)}
    category_to_idx['Unknown'] = 0
    
    valid_user_ids = set(df_user_chars['user_id'].astype(str))
    filtered_purchase_map = {uid: hist for uid, hist in user_purchase_map.items() 
                           if str(uid) in valid_user_ids}
    
    df_user_chars_indexed = df_user_chars.set_index('user_id')
    
    feature_list = []
    user_ids = list(filtered_purchase_map.keys())
    
    # changable batch size depending on your pc spec
    batch_size = 1
    for i in tqdm(range(0, len(user_ids), batch_size), desc="Processing user batches", ncols=80):
        batch_users = user_ids[i:i+batch_size]
        batch_features = process_user_batch(
            batch_users, filtered_purchase_map, df_user_chars_indexed,
            product_to_category, category_to_idx, n_latest
        )
        feature_list.extend(batch_features)
    
    return pd.DataFrame(feature_list)

def process_user_batch(user_ids, purchase_map, user_chars_df, product_to_category, category_to_idx, n_latest):
    """Process a batch of users efficiently."""
    batch_features = []
    
    for user_id in user_ids:
        try:
            user_char = user_chars_df.loc[user_id]
        except KeyError:
            continue
            
        purchase_history = purchase_map[user_id]
        
        features = {
            'user_id': user_id,
            'avg_item_avg_rating': user_char.get('avg_item_avg_rating', 0),
            'total_purchases': user_char.get('total_purchases', 0),
            'distinct_categories': user_char.get('distinct_categories', 0),
            'norm_entropy': user_char.get('norm_entropy', 0),
            'importance': user_char.get('importance', 0),
            'account_age': max(user_char.get('last_review_time', 0) - user_char.get('first_review_time', 0), 0)
        }
        
        latest_purchases = purchase_history[-n_latest:] if len(purchase_history) >= n_latest else purchase_history
        
        categories = [product_to_category.get(pid, 'Unknown') for pid in latest_purchases]
        category_indices = [category_to_idx.get(cat, 0) for cat in categories]
        
        category_indices.extend([0] * (n_latest - len(category_indices)))
        
        for i in range(n_latest):
            features[f'latest_purchase_{i+1}_category'] = category_indices[i]
        
        category_counts = np.bincount(category_indices, minlength=len(category_to_idx))
        
        for cat, idx in category_to_idx.items():
            if cat == 'Unknown':
                features['category_count_unknown'] = category_counts[0]
            else:
                features[f'category_count_{cat}'] = category_counts[idx]
        
        features['most_frequent_category'] = np.argmax(category_counts)
        
        batch_features.append(features)
    
    return batch_features

def main():
    ## Load in all the data
    data_io.resync_registry()

    data_loader = DataLoader()

    # categories = data_loader.load_categories_from_file('data/raw/all_categories.txt')
    # categories = data_loader.ensure_categories_downloaded(categories)

    # print(categories)

    categories = ['All_Beauty'] # 'Amazon_Fashion'
    
    category_one_hot_map = data_loader.build_one_hot(categories)
    
    ## Create user history across multiple categories
    data_pipeline = DataPipeLine()
    
    user_purchase_map = data_pipeline.create_user_history(categories)
    df_user_chars = data_pipeline.build_user_characteristics(categories)
    df_item_features = data_pipeline.get_all_top_item_features(categories)
    
    # print("Building feature vectors...")
    feature_df = build_top_n_features(
        category_one_hot_map=category_one_hot_map,
        user_purchase_map=user_purchase_map,
        df_user_chars=df_user_chars,
        df_item_features=df_item_features,
        n_latest=5
    )
    
    # Create training samples
    X, y, training_df = create_training_samples(
        feature_df, user_purchase_map, df_item_features, category_one_hot_map
    )

    print("\nSample feature vector:")
    print("Feature:", X.head(1).T)
    print("Label:", y.head(1).T)
    
    # Save data
    feature_df.to_parquet('data/global/user_feature_vectors.parquet', index=False)
    training_df.to_parquet('data/global/training_samples.parquet', index=False)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    feature_vectors_path = os.path.join(project_root, 'data/global/user_feature_vectors.parquet')
    training_samples_path = os.path.join(project_root, 'data/global/training_samples.parquet')

    data_io.upload_to_drive(feature_vectors_path)
    data_io.upload_to_drive(training_samples_path)
    
    print("Feature vectors and training samples saved!")


if __name__ == "__main__":
    main()