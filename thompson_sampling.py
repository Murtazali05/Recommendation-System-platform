import random

import matplotlib.pyplot as plt

path = 'datasets/data/jesterfinal151cols.csv'
dataset = get_jester_data(path)

# Thomson sampling
number_of_users = 10101
arm_count = 4
models_selected = []
numbers_of_rewards_1 = [0] * arm_count
numbers_of_rewards_0 = [0] * arm_count
total_reward = 0
current_reward = [0] * number_of_users

for t in range(0, number_of_users):
    model = 0
    max_random = 0
    for i in range(0, arm_count):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            model = i
    models_selected.append(model)
    reward = dataset[t, model]  # вместо dataset должен быть
    if reward == 1:
        numbers_of_rewards_1[model] = numbers_of_rewards_1[model] + 1
    else:
        numbers_of_rewards_0[model] = numbers_of_rewards_0[model] + 1

    total_reward = total_reward + reward
    current_reward[t] = total_reward

# Visualising
plt.hist(models_selected)
plt.title('Thomson Sampling: Histogram of jokes selections')
plt.xlabel('Jokes')
plt.ylabel('Number of times each joke was selected')
plt.show()


plt.plot([i for i in range(0, arm_count)], current_reward)
plt.xlabel("Step (s)")
plt.ylabel("Reward")
plt.show()
