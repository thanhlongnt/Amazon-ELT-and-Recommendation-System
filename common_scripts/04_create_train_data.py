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
class DataLoader:
    def ensure_categories_downloaded(self, categories:List[str]):
        for cat in categories:
            try:
                data_io.ensure_local_path(f"data/processed/{cat}/top_item_features_{cat}.parquet")
                data_io.ensure_local_path(f"data/processed/{cat}/top_user_features_{cat}.parquet")
                data_io.ensure_local_path(f"data/processed/{cat}/top_user_reviews_{cat}.parquet")
            except:
                print(f"top user {cat} not processed yet")
                with open('data/processed/missing_categories.txt', 'a') as f:
                    f.write(f"{cat}\n")


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
    


def build_top_n_features(category_one_hot_map, user_purchase_map, df_user_chars, df_item_features, n_latest=10):
    """
    Build feature vectors combining user characteristics, item features, and n latest purchases.
    
    Args:
        category_one_hot_map: Dict mapping categories to one-hot indices
        user_purchase_map: Dict mapping user_id to list of purchased product_ids (sorted by time)
        df_user_chars: DataFrame with user characteristics
        df_item_features: DataFrame with item features including categories
        n_latest: Number of latest purchases to include
    
    Returns:
        DataFrame with feature vectors for each user
    """
    
    # Create a mapping from product_id to category for quick lookup
    product_to_category = df_item_features.set_index('product_id')['category'].to_dict()
    
    # Get number of categories for one-hot encoding
    num_categories = len(category_one_hot_map)
    
    feature_vectors = []
    
    for user_id in tqdm(list(user_purchase_map.keys())[:100], desc="Building feature vectors", ncols=80):
        # Get user characteristics
        user_char = df_user_chars[df_user_chars['user_id'] == user_id]
        if user_char.empty:
            continue
            
        user_char = user_char.iloc[0]
        
        features = {
            'user_id': user_id,
            'avg_item_avg_rating': user_char.get('avg_item_avg_rating', 0),
            'total_purchases': user_char.get('total_purchases', 0),
            'distinct_categories': user_char.get('distinct_categories', 0),
            'entropy': user_char.get('entropy', 0),
            'norm_entropy': user_char.get('norm_entropy', 0),
            'importance': user_char.get('importance', 0)
        }
        
        if 'first_review_time' in user_char:
            features['first_review_time'] = user_char['first_review_time']
        if 'last_review_time' in user_char:
            features['last_review_time'] = user_char['last_review_time']
            
        purchase_history = user_purchase_map[user_id]
        
        latest_purchases = purchase_history[-n_latest:] if len(purchase_history) >= n_latest else purchase_history
        
        category_counts = np.zeros(num_categories + 1)  # +1 for unknown categories
        
        for i in range(n_latest):
            if i < len(latest_purchases):
                product_id = latest_purchases[-(i+1)]
                category = product_to_category.get(product_id, 'Unknown')
                
                cat_idx = category_one_hot_map.get(category, 0)  # 0 for unknown
                
                features[f'latest_purchase_{i+1}_category'] = cat_idx
                
                category_counts[cat_idx] += 1
            else:
                features[f'latest_purchase_{i+1}_category'] = 0
        
        # Add category frequency features
        for category, idx in category_one_hot_map.items():
            features[f'category_count_{category}'] = category_counts[idx]
        features['category_count_unknown'] = category_counts[0]
        
        # Add aggregate features
        features['total_category_diversity'] = np.sum(category_counts > 0)
        features['most_frequent_category'] = np.argmax(category_counts)
        features['purchase_history_length'] = len(purchase_history)
        
        # Add recency features (how recent each category was purchased)
        category_recency = {}
        for i, product_id in enumerate(reversed(purchase_history)):
            category = product_to_category.get(product_id, 'Unknown')
            if category not in category_recency:
                category_recency[category] = i + 1  # 1 = most recent
        
        for category, idx in category_one_hot_map.items():
            features[f'recency_{category}'] = category_recency.get(category, len(purchase_history) + 1)
        features['recency_unknown'] = category_recency.get('Unknown', len(purchase_history) + 1)
        
        feature_vectors.append(features)
    
    feature_df = pd.DataFrame(feature_vectors)
    
    print(f"Built feature vectors for {len(feature_df)} users")
    print(f"Feature vector dimension: {len(feature_df.columns)}")
    print(f"Feature columns: {list(feature_df.columns)}")
    
    return feature_df

def create_training_samples(feature_df, user_purchase_map, df_item_features, category_one_hot_map):
    """
    Create training samples with features and target labels.
    For each user, create samples where we predict the next category given purchase history up to that point.
    """
    
    product_to_category = df_item_features.set_index('product_id')['category'].to_dict()
    training_samples = []
    
    for _, user_row in feature_df.iterrows():
        user_id = user_row['user_id']
        purchase_history = user_purchase_map[user_id]
        
        # Create multiple training samples per user (predicting each subsequent purchase)
        for i in range(1, len(purchase_history)):  # Start from 1 to have at least 1 purchase in history
            # Use purchase history up to index i-1 to predict purchase at index i
            target_product = purchase_history[i]
            target_category = product_to_category.get(target_product, 'Unknown')
            target_label = category_one_hot_map.get(target_category, 0)
            
            # Create feature vector based on history up to this point
            sample_features = user_row.copy()
            sample_features['target_category'] = target_label
            sample_features['sample_index'] = i
            
            training_samples.append(sample_features)
    
    training_df = pd.DataFrame(training_samples)
    
    # Separate features and targets
    feature_columns = [col for col in training_df.columns if col not in ['user_id', 'target_category', 'sample_index']]
    X = training_df[feature_columns]
    y = training_df['target_category']
    
    print(f"Created {len(training_df)} training samples")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:")
    print(y.value_counts())
    
    return X, y, training_df

def main():
    ## Load in all the data
    data_loader = DataLoader()

    # categories = data_loader.load_categories_from_file('data/raw/all_categories.txt')

    categories = ['All_Beauty', 'Amazon_Fashion']
    
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
    print("\nCreating training samples...")
    X, y, training_df = create_training_samples(
        feature_df, user_purchase_map, df_item_features, category_one_hot_map
    )

    print("\nSample feature vector:")
    print("Feature:", X.head(1).T)
    print("Label:", y.head(1).T)
    
    # Save the processed data
    feature_df.to_parquet('data/processed/user_feature_vectors.parquet', index=False)
    training_df.to_parquet('data/processed/training_samples.parquet', index=False)
    
    print("Feature vectors and training samples saved!")


if __name__ == "__main__":
    main()