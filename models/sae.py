import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable


# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self, nb_users, nb_movies):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

    # Training the SAE
    def train_model(self, nb_epoch, train_set, criterion, optimizer, nb_users, nb_movies):
        for epoch in range(1, nb_epoch + 1):
            train_loss = 0
            s = 0.
            for id_user in range(train_set.shape[0]):
                input = Variable(train_set[id_user]).unsqueeze(0)
                target = input.clone()
                if torch.sum(target.data > 0) > 0:
                    output = self(input)
                    target.require_grad = False
                    output[target == 0] = 0
                    loss = criterion(output, target)
                    mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)
                    loss.backward()
                    train_loss += np.sqrt(loss.cpu().data * mean_corrector)
                    s += 1.  # users that gave at least one non-zero rating
                    optimizer.step()
            print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s))

            if epoch % 25 == 0:
                torch.save(self.state_dict(), 'sae_checkpoints/sae_{}.weights'.format(epoch))

    # Testing the SAE
    def test(self, training_set, test_set, criterion, nb_users, nb_movies):
        test_loss = 0
        s = 0.
        for id_user in range(nb_users):
            input = Variable(training_set[id_user]).unsqueeze(0)
            target = Variable(test_set[id_user]).unsqueeze(0)
            if torch.sum(target.data > 0) > 0:
                output = self(input)
                target.require_grad = False
                output[target == 0] = 0
                loss = criterion(output, target)
                mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)
                test_loss += np.sqrt(loss.data * mean_corrector)
                s += 1.
        print('test loss: ' + str(test_loss / s))

    # Predictions
    def predict_recommendations(self, unseen_set, user_id=None, N=5):
        if user_id is None:
            user_id = random.randint(0, unseen_set.shape[0])

        print("Predicting Recommendations for user_id: {}".format(user_id))
        input = Variable(unseen_set[user_id]).unsqueeze(0)
        output = self(input)

        out_df = pd.DataFrame(output.cpu().detach().numpy()[0], columns=['predicted_ratings'])
        out_df['ground_truth'] = input.cpu().detach().numpy()[0]

        # filtering unwatched movies and sorting according to the predicted rating
        rec_df = out_df[out_df['ground_truth'] == 0].sort_values(by='predicted_ratings', ascending=False)
        recs = [x + 1 for x in rec_df[:4 * N].index]
        return random.sample(recs, k=N)
