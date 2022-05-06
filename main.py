# To restrict number of CPU cores uncomment the lines of code below.

# import os
# os.environ["OMP_NUM_THREADS"] = "24"
# os.environ["MKL_NUM_THREADS"] = "24"
# os.environ["NUMBA_NUM_THREADS"] = "24"

import numpy as np
import pandas as pd
from tqdm import tqdm

from evaluation import downvote_seen_items, topn_recommendations
from scipy.linalg import sqrtm
#from IPython.utils import io

from Modules import read_amazon_data, full_preproccessing, make_prediction, model_evaluate
from Random_MP import build_random_model, random_model_scoring, build_popularity_model, popularity_model_scoring
from NormPureSVD_EASEr import build_svd_model, svd_model_scoring, easer, easer_scoring
from CoFFee_LaTTe import full_pipeline, get_similarity_matrix
import sys

names = ["Movielens_1M", "Movielens_10M", "Video_Games", "CDs_and_Vinyl", "Electronics", "Video_Games_nf"]

if len(sys.argv) > 1:
   data_name = sys.argv[1]
   assert data_name in names, f"Name of dataset must be one the following {names}"
else:
   data_name = "Movielens_1M"

print(f"Starting tuning models for dataset {data_name}.\n")

if (data_name != "Electronics"):
   q = 0.8
else:
   q = 0.95

if data_name == "Movielens_1M":
   data = None
   name = "ml-1m.zip"
elif data_name == "Movielens_10M":
   data = None
   name = "ml-10m.zip"
elif data_name == "Video_Games_nf":
   data = pd.read_csv("ratings_Video_Games.csv")
   name = None
else:
   data = read_amazon_data(name = data_name)
   data.rename(columns = {'reviewerID' : 'userid', 'asin' : 'movieid', "overall" : "rating", "unixReviewTime" : "timestamp"}, inplace = True) 
   name = None


training, testset_valid, holdout_valid, testset, holdout, data_description, data_index = full_preproccessing(data, name, q)

print("Tuning PureSVD on progress...")

rank_grid = []
for i in range(5, 9):
    rank_grid.append(2 * 2 ** i)
    rank_grid.append(3 * 2 ** i)

rank_grid = np.array(rank_grid)
f_grid = np.linspace(0, 2, 21)

hr_tf = {}
mrr_tf = {}
C_tf = {}
for f in tqdm(f_grid):
    svd_config = {'rank': rank_grid[-1], 'f': f}
    svd_params = build_svd_model(svd_config, training, data_description)
    for r in rank_grid:
        svd_scores = svd_model_scoring(svd_params[:, :r], testset_valid, data_description)
        downvote_seen_items(svd_scores, testset_valid, data_description)
        svd_recs = topn_recommendations(svd_scores, topn=10)
        hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C = model_evaluate(svd_recs, holdout_valid, data_description, alpha=3, topn=10, dcg=False)
        hr_tf[f'r={r}, f={f:.2f}'] = hr
        mrr_tf[f'r={r}, f={f:.2f}'] = mrr
        C_tf[f'r={r}, f={f:.2f}'] = C

print("Validation tuning result:")
print("HR")
hr_sorted = sorted(hr_tf, key=hr_tf.get, reverse=True)
for i in range(1):
    print(hr_sorted[i], hr_tf[hr_sorted[i]])

print("MRR")
mrr_sorted = sorted(mrr_tf, key=mrr_tf.get, reverse=True)
for i in range(1):
    print(mrr_sorted[i], mrr_tf[mrr_sorted[i]])

print("MCC")
C_sorted = sorted(C_tf, key=C_tf.get, reverse=True)
for i in range(1):
    print(C_sorted[i], C_tf[C_sorted[i]])


print("Evaluation on testset (Random, MP, SVD) in progress...")


data_description["test_users"] = holdout[data_index['users'].name].drop_duplicates().values
data_description["n_test_users"] = holdout[data_index['users'].name].nunique()

print("Random:")

rnd_params = build_random_model(training, data_description)
rnd_scores = random_model_scoring(rnd_params, None, data_description)
downvote_seen_items(rnd_scores, testset, data_description)
_ = make_prediction(rnd_scores, holdout, data_description, mode="Test")
print()

print("MP:")

pop_params = build_popularity_model(training, data_description)
pop_scores = popularity_model_scoring(pop_params, None, data_description)
downvote_seen_items(pop_scores, testset, data_description)
_ = make_prediction(pop_scores, holdout, data_description, mode="Test")
print()


print('Normalized PureSVD:')

for_hr = sorted(hr_tf, key=hr_tf.get, reverse=True)[0]
for_mrr = sorted(mrr_tf, key=mrr_tf.get, reverse=True)[0]
for_mc = sorted(C_tf, key=C_tf.get, reverse=True)[0]

svd_config_hr = {'rank': int(for_hr.split(",")[0][2:]), 'f': float(for_hr.split(",")[1][3:])}
svd_config_mrr = {'rank': int(for_mrr.split(",")[0][2:]), 'f': float(for_mrr.split(",")[1][3:])}
svd_config_mc = {'rank': int(for_mc.split(",")[0][2:]), 'f': float(for_mc.split(",")[1][3:])}

svd_configs = [(svd_config_hr, "Tuned by HR"), (svd_config_mrr, "Tuned by MRR"), (svd_config_mc, "Tuned by MCC")]

for svd_config in svd_configs:
    print(svd_config)
    svd_params = build_svd_model(svd_config[0], training, data_description)
    svd_scores = svd_model_scoring(svd_params, testset, data_description)
    downvote_seen_items(svd_scores, testset, data_description)

    _ = make_prediction(svd_scores, holdout, data_description, mode="Test")

print("Evaluation on testset (Random, MP, SVD) ended.\n")

factor = float(for_mc.split(",")[1][3:])

print("CoFFee tuning in progress...")    

config = {
    "scaling": 1,
    "mlrank": (30, 30, data_description['n_ratings']),
    "n_ratings": data_description['n_ratings'],
    "num_iters": 5,
    "params": None,
    "randomized": True,
    "growth_tol": 1e-4,
    "seed": 42
}

data_description["test_users"] = holdout_valid[data_index['users'].name].drop_duplicates().values
data_description["n_test_users"] = holdout_valid[data_index['users'].name].nunique()

attention_matrix = np.eye(data_description["n_ratings"])
full_pipeline(config, training, data_description, testset_valid, holdout_valid, testset, holdout, attention_matrix=attention_matrix, factor = factor)

print("LaTTe tuning in progress...\n")

modes = [ "linear", "sq3", "sigmoid", "arctan"]

for mode in modes:
    print(f"For similarity matrix '{mode}'' tuning...")
    similarity_matrix = get_similarity_matrix(mode, data_description["n_ratings"])
    attention_matrix = sqrtm(similarity_matrix).real
    full_pipeline(config, training, data_description, testset_valid, holdout_valid, testset, holdout, attention_matrix=attention_matrix, factor = float(for_mc.split(",")[1][3:]))
    print("_____________________________________________________")

print("LaTTe tuning ended.\n")

print("EASEr tuning in progress...\n")

lambda_grid = np.arange(50, 1000, 50)

hr_tf = {}
mrr_tf = {}
C_tf = {}

for lmbda in tqdm(lambda_grid):
    easer_params = easer(training, data_description, lmbda=lmbda)
    easer_scores = easer_scoring(easer_params, testset_valid, data_description)
    downvote_seen_items(easer_scores, testset_valid, data_description)
    easer_recs = topn_recommendations(easer_scores, topn=10)
    hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C = model_evaluate(easer_recs, holdout_valid, data_description, alpha=3, topn=10, dcg=False)
    hr_tf[lmbda] = hr
    mrr_tf[lmbda] = mrr
    C_tf[lmbda] = C

print("Validation tuning result:")

print("HR")
hr_sorted = sorted(hr_tf, key=hr_tf.get, reverse=True)
for i in range(1):
    print(hr_sorted[i], hr_tf[hr_sorted[i]])

print("MRR")
mrr_sorted = sorted(mrr_tf, key=mrr_tf.get, reverse=True)
for i in range(1):
    print(mrr_sorted[i], mrr_tf[mrr_sorted[i]])

print("MCC")
C_sorted = sorted(C_tf, key=C_tf.get, reverse=True)
for i in range(1):
    print(C_sorted[i], C_tf[C_sorted[i]])

print("Evaluation on testset (EASEr) in progress...")

data_description["test_users"] = holdout[data_index['users'].name].drop_duplicates().values
data_description["n_test_users"] = holdout[data_index['users'].name].nunique()

easer_params = easer(training, data_description, lmbda=C_sorted[i])
easer_scores = easer_scoring(easer_params, testset, data_description)
downvote_seen_items(easer_scores, testset, data_description)

_ = make_prediction(easer_scores, holdout, data_description, mode='Test')
