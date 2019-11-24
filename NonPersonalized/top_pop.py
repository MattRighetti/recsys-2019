import numpy as np

class TopPop(object):
    def __init__(self):
        self.URM_train = None
        self.popularItems = None

    def fit(self, URM_train):
        self.URM_train = URM_train.tocsc()

        self.popularItems = np.ediff1d(self.URM_train.indptr)
        ind = np.argpartition(self.popularItems, -self.URM_train.shape[1])[-self.URM_train.shape[1]:]

        # We are not interested in sorting the popularity value,
        # but to order the items according to it
        self.popularItems = np.flip(ind[np.argsort(self.popularItems[ind])])

    def recommend(self, user_id, at=10, remove_seen=True):
        if remove_seen:
            unseen_items_mask = np.in1d(self.popularItems, self.URM_train[user_id].indices,
                                        assume_unique=True, invert=True)

            unseen_items = self.popularItems[unseen_items_mask]

            recommended_items = unseen_items[0:at]

        else:
            recommended_items = self.popularItems[0:at]

        return recommended_items