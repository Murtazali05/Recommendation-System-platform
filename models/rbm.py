import random

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


# Creating the architecture of neural network
class RBM(nn.Module):

    def __init__(self, nv, nh):
        super().__init__()
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)

    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

    def train_model(self, nb_epochs, batch_size, train_set):
        nb_users = train_set.shape[0]
        for epoch in range(1, nb_epochs + 1):
            # import pdb;pdb.set_trace()
            train_loss = 0
            s = float(0)
            for id_user in range(0, nb_users - batch_size, batch_size):
                vk = train_set[id_user:id_user + batch_size]
                v0 = train_set[id_user:id_user + batch_size]
                ph0, _ = self.sample_h(v0)
                for k in range(10):
                    _, hk = self.sample_h(vk)
                    _, vk = self.sample_v(hk)
                    vk[v0 < 0] = v0[v0 < 0]
                phk, _ = self.sample_h(vk)
                self.train(v0, vk, ph0, phk)
                train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
                s += 1.
                # print(train_loss)

            print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s))

    def test(self, train_set, test_set):
        test_loss = 0
        s = float(0)
        nb_users = test_set.shape[1]
        for id_user in range(nb_users):

            v = train_set[id_user:id_user + 1]
            vt = test_set[id_user:id_user + 1]
            if len(vt[vt >= 0]) > 0:
                _, h = self.sample_h(v)
                _, v = self.sample_v(h)
                test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
                s += 1.
        print(' loss: ' + str(test_loss / s))

    def predict_recommendations(self, unseen_set, user_id=None, N=5):
        if not user_id:
            user_id = random.randint(0, unseen_set.shape[0])
        print("predicting recommendations for user: {}".format(user_id))
        v = unseen_set[user_id:user_id + 1]
        _, h = self.sample_h(v)
        _, v = self.sample_v(h)

        out_df = pd.DataFrame(v.numpy()[0], columns=['predicted_ratings'])
        out_df['ground_truth'] = unseen_set[user_id:user_id + 1].numpy()[0]

        # filtering unwatched movies and sorting according to the predicted rating
        rec_df = out_df[out_df['ground_truth'] == -1][out_df['predicted_ratings'] == 1].sort_values(
            by='predicted_ratings',
            ascending=False)
        recs = [x + 1 for x in rec_df[:4 * N].index]
        return random.sample(recs, k=N)
