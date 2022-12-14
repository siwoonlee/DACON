from laboratory.transaction_classification.utils.vectorizer import OOV_INDEX


def min_max_norm_title_vec(vec):
    return (vec - 0) / (OOV_INDEX + 1 - 0)
