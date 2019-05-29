import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(5,5))
ax = plt.gca()

plt.plot([1,1],[0,1], color='red', linewidth=2)
plt.plot([1,2],[2,2], color='red', linewidth=2)
plt.plot([2,2],[2,1], color='red', linewidth=2)
plt.plot([2,3],[1,1], color='red', linewidth=2)

plt.text(0.5, 2.5, 'S0', size=14, ha='center')
plt.text(1.5, 2.5, 'S1', size=14, ha='center')
plt.text(2.5, 2.5, 'S2', size=14, ha='center')
plt.text(0.5, 1.5, 'S3', size=14, ha='center')
plt.text(1.5, 1.5, 'S4', size=14, ha='center')
plt.text(2.5, 2.5, 'S5', size=14, ha='center')
plt.text(0.5, 0.5, 'S6', size=14, ha='center')
plt.text(1.5, 0.5, 'S7', size=14, ha='center')
plt.text(2.5, 0.5, 'S8', size=14, ha='center')
plt.text(0.5, 2.3, 'START', ha='center')
plt.text(2.5, 0.3, 'GOAL', ha='center')

ax.set_xlim(0, 3)
ax.set_ylim(0, 3)
plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off',right='off',labelleft='off')

line, =  ax.plot([0.5],[2.5], marker="o", color='g', markersize=60)

plt.show()

theta_0 = np.array(
            [[np.nan, 1, 1, np.nan], #S0
            [np.nan, 1, np.nan, 1],  #S1
            [np.nan, np.nan, 1, 1], #S2
            [1, 1 , 1, np.nan], #S3
            [np.nan, np.nan, 1, 1], #S4
            [1, np.nan, np.nan, np.nan], #S5
            [1, np.nan, np.nan, np.nan], #S6
            [1, 1, np.nan, np.nan] #S7
            ])

def simple_convert_into_pi_from_theta(theta):
    [m, n] = theta.shape
    pi = np.zeros((m, n))
    for i in range(0, m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])
    pi = np.nan_to_num(pi)
    return pi

pi_0 = simple_convert_into_pi_from_theta(theta_0)

print(pi_0)


def get_next_s(pi, s):
    direction = ["up","right","down","left"]
    next_direction = np.random.choice(direction,p=pi[s,:])


    if next_direction =='up':
        s_next = s -3
    elif next_direction == 'right':
        s_next = s + 1
    elif next_direction == 'down':
        s_next = s + 3
    if next_direction == 'left':
        s_next = s -1

    return s_next



def get_action_and_next_s(pi, s):
    direction = ["up", "right", "down", "left"]
    # pi[s,:]の確率に従って、directionが選択される
    next_direction = np.random.choice(direction, p=pi[s, :])

    if next_direction == "up":
        action = 0
        s_next = s - 3  # 上に移動するときは状態の数字が3小さくなる
    elif next_direction == "right":
        action = 1
        s_next = s + 1  # 右に移動するときは状態の数字が1大きくなる
    elif next_direction == "down":
        action = 2
        s_next = s + 3  # 下に移動するときは状態の数字が3大きくなる
    elif next_direction == "left":
        action = 3
        s_next = s - 1  # 左に移動するときは状態の数字が1小さくなる

    return [action, s_next]


def goal_maze(pi):
    s = 0
    state_history = [0]

    while(1):
        next_s = get_next_s(pi, s)
        state_history.append(next_s)

        if next_s == 8:
            break
        else:
            s = next_s
    
    return state_history

s_a_history = goal_maze(pi_0)

def goal_maze_ret_s_a(pi):
    s = 0  # スタート地点
    s_a_history = [[0, np.nan]]  # エージェントの移動を記録するリスト

    while (1):  # ゴールするまでループ
        [action, next_s] = get_action_and_next_s(pi, s)
        s_a_history[-1][1] = action
        # 現在の状態（つまり一番最後なのでindex=-1）の行動を代入

        s_a_history.append([next_s, np.nan])
        # 次の状態を代入。行動はまだ分からないのでnanにしておく

        if next_s == 8:  # ゴール地点なら終了
            break
        else:
            s = next_s

    return s_a_history

s_a_history = goal_maze_ret_s_a(pi_0)


print(s_a_history)
print("迷路を解くのにかかったのは",len(s_a_history)-1,"ステップです。")


def update_theta(theta, pi, s_a_histroy):
    eta = 0.1
    T = len(s_a_histroy)-1

    [m,n] = theta.shape
    delta_theta = theta.copy() # Δθ

    for i in range(0, m):
        for j in range(0, n):
            if not (np.isnan(theta[i, j])):
                SA_i = [SA for SA in s_a_history if SA[0] == i]
                # 履歴から状態iのものを取り出すリスト内包表記です

                SA_ij = [SA for SA in s_a_history if SA == [i, j]]
                # 状態iで行動jをしたものを取り出す

                N_i = len(SA_i)  # 状態iで行動した総回数
                N_ij = len(SA_ij)  # 状態iで行動jをとった回数

                N_i = len(SA_i)
                N_ij = len(SA_ij)
                delta_theta[i,j] = (N_ij + pi[i, j]*N_i) / T
    
    new_theta = theta + eta * delta_theta
    return new_theta


new_theta = update_theta(theta_0, pi_0, s_a_history)
pi = simple_convert_into_pi_from_theta(new_theta)
print(pi)
