import pandas as pd
from sklearn.model_selection import train_test_split
import os

def create_train_test_splits():
    # 1. Indlæs data fra de eksisterende filer
    terrain_data = pd.read_csv("ground_truth.csv")
    crowns_data = pd.read_csv("tiles_crowns.csv")
    
    # 2. Flet de to datasæt
    merged_data = pd.merge(
        terrain_data,
        crowns_data,
        on=['image_id', 'x', 'y'],
        how='inner'  # Kun rækker der findes i begge filer
    )
    
    # 3. Split 80/20 - brug kun unikke billed-ID'er
    unique_ids = merged_data['image_id'].unique()
    train_ids, test_ids = train_test_split(
        unique_ids, 
        test_size=0.2, 
        random_state=42
    )
    
    # 4. Gem de nye CSV-filer
    merged_data[merged_data['image_id'].isin(train_ids)].to_csv(
        "ground_truth_train_split.csv",
        index=False
    )
    merged_data[merged_data['image_id'].isin(test_ids)].to_csv(
        "ground_truth_test_split.csv",
        index=False
    )
    
    print(f"✅ Oprettet train split: ground_truth_train_split.csv ({len(train_ids)} billeder)")
    print(f"✅ Oprettet test split: ground_truth_test_split.csv ({len(test_ids)} billeder)")

if __name__ == "__main__":
    # Vis aktuel arbejdsmappe for debugging
    print(f"Arbejder i mappen: {os.getcwd()}")
    print(f"Filer i mappen: {os.listdir()}")
    
    # Kør split-funktionen
    create_train_test_splits()