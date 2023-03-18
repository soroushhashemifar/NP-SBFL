import pickle


with open("pickles/Model_2_tarantula_objects.pickle", 'rb') as handle:
    decision_birch = pickle.load(handle)

print(decision_birch)

# with open("pickles/synthesized_dataset_Model_2_tarantula.pickle", 'rb') as handle:
#     decision_birch = pickle.load(handle)

# d, dp, l = decision_birch[0]
# print(dp[d == dp])