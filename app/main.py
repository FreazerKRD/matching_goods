import uvicorn
import faiss
import os
import sys
import numpy as np
import pandas as pd
import catboost as cb
from sklearn.preprocessing import StandardScaler
from config import DATA_PATH
from faiss import read_index
from fastapi import FastAPI
from typing import Union
from joblib import dump, load

app = FastAPI()
dim = 72
idx_l2 = None
cb_model = None
scaler = None
scaled_vectors = pd.DataFrame()
base_index = {}
# Number of neighbors to search
k_neighbors = 500
# number of neighbors to compare with target
n_neighbors = 5

# Function to parse incoming string to list of float
def parse_string(vec: str) -> list[float]:
    l = vec.split(",")
    if len(l) != dim:
        return None
    return [float(el) for el in l]

# Function to select candidates with maximum prediction probabilty
def find_max_indices(arr: np.array, k: int, n: int) -> np.array:
    max_indices = np.empty((0, n), dtype=int)
    for i in range(0, len(arr), k):
        sub_arr = arr[i:i+k]
        max_values = np.argpartition(sub_arr, -n)[-n:]
        max_indices = np.vstack((max_indices, max_values))
    return max_indices

@app.on_event("startup")
def start() -> None:
    global idx_l2
    global cb_model
    global scaler
    global scaled_vectors
    global base_index
    
    try:
        # Load index from cache
        idx_l2 = read_index(os.path.join(DATA_PATH, "idx_l2.index"))
        # number of clusters to check
        idx_l2.nprobe = 256
        # Load Catboost model
        cb_model = cb.CatBoostClassifier()
        cb_model.load_model(os.path.join(DATA_PATH, "cb_final_model"))
        # Load scaled_vectors and base_index
        scaled_vectors = pd.read_csv("app/data/scaled_vectors.csv", index_col=0)
        base_index = np.load('app/data/base_index.npy', allow_pickle='TRUE').item()
        # Load scaler
        scaler=load('app/data/scaler.bin')
        print("System files loaded successfully")
    except:
        sys.exit("Missing required files")

@app.get("/")
def main() -> dict:
    return {"status": "OK", "Message": "App working, use /knn or /add query"}

@app.get("/knn")
def match(item: Union[str, None] = None) -> dict:
    global idx_l2
    global cb_model
    global scaler
    global scaled_vectors
    global base_index

    if item is None:
        return {"status": "Fail", "Message": "No data recieved"}
    
    vec = np.array(parse_string(item)).reshape(1, -1)

    # Scale vector
    scaled_vec = pd.DataFrame(scaler.transform(vec))

    # Clean memory
    del vec

    # Take distancies and neighbors indexes
    dist, idx = idx_l2.search(np.ascontiguousarray(scaled_vec.values).astype('float32'), k_neighbors)

    # Make index as np.array
    idx = np.array([[base_index[_] for _ in el] for el in idx], dtype=object)
    
    # Take features for candidates
    candidate_features = scaled_vectors.loc[idx.flatten()].values
    object_features = scaled_vec.values
    
    # Reshape distancies
    reshaped_dist = dist.reshape(-1,1)

    # Repeating data for all candidates
    repeated_object = np.repeat(object_features, k_neighbors, axis=0)
    
    # Horizontal merge of arrays
    cb_features = np.hstack((reshaped_dist, candidate_features, repeated_object))
    
    # Make predictions on validation chunk data
    cb_preds = cb_model.predict_proba(cb_features)[:,1]
    
    # Set number of candidates and select their indices 
    best_candidates = find_max_indices(cb_preds.flatten(), k_neighbors, n_neighbors)
    
    # Select names of candidates via their indices
    candidates_idx = idx[np.arange(len(idx))[:, None], best_candidates]
    
    # Clear memory
    del object_features
    del reshaped_dist
    del repeated_object
    del candidate_features
    del dist
    del scaled_vec

    return {"status": "OK", "data": candidates_idx.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8031)