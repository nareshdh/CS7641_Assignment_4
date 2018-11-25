import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

colors = ['r', 'b', 'g', 'k', 'm', 'c', 'y', '#eeefff', '.75', '.25']

OPTIMAL_Q_EASY = 'Easy Q-Learning L0.1 q0.0 E0.1.csv'
OPTIMAL_Q_HARD = 'Hard Q-Learning L0.1 q0.0 E0.1.csv'

def get_df(path, headerVal=0):
    return pd.read_csv('./results/' + path, header=headerVal)

def get_compdf(difficulty):
	db = pd.DataFrame()
	list_ = []
	if difficulty.title() == 'Easy':
		for f in glob.glob('./results/Easy Q-*'):
			df = pd.read_csv(f, index_col=None, header=0)
			list_.append(df)
	if difficulty.title() == 'Hard':
		for f in glob.glob('./results/Hard Q-*'):
			df = pd.read_csv(f, index_col=None, header=0)
			list_.append(df)
	return pd.concat(list_)

def find_best_q_learning(difficulty):
    bestResults = list()

    for lr in [0.1,0.5,0.9]:
        for qInit in [-100,0,100]:
            for epsilon in [0.1,0.3,0.5,0.7,0.9]:
                params = 'L' + str(lr) + ' q' + str(qInit) + '.0 E' + str(epsilon)
                path = difficulty.title() + ' Q-Learning ' + params + '.csv'
                # print(path)
                df = get_df(path)

                df = df.sort_values('iter', ascending=False)
                max_reward = df['reward'].idxmax()
                max_row = df.ix[max_reward]
                max_row['params'] = params
                bestResults.append(max_row)

    df = pd.DataFrame(bestResults)
    df = df.sort_values('iter', ascending=False)
    df = df.sort_values('reward', ascending=False)
    print(df)

    count = 0
    for index, row in df.iterrows():
        count = count + 1
        if count > 10:
            continue

        sig = 4
        sigString = '{:0.' + str(sig) + 'f}'
        vals = map(lambda x: sigString.format(x).replace(".0000", ''), [ row['iter'], row['time'], row['reward'], row['steps'], row['convergence'] ])
        print('\\textbf{' + row['params'] + '} & ' + ' & '.join(vals) + ' \\\\ \\hline')


def print_value_results(difficulty, policy):
    path = difficulty + '/' + difficulty.title() + ' ' + policy + '.csv' 
    print(path)
    df = get_df(path)

    count = 0
    for index, row in df.iterrows():
        if count == 0 or (count + 1) % 5 == 0 or count + 1 == len(df):
            sig = 4
            sigString = '{:0.' + str(sig) + 'f}'
            vals = map(lambda x: sigString.format(x).replace(".0000", ''), [ row['iter'], row['time'], row['reward'], row['steps'], row['convergence'] ])
            print(' & '.join(vals) + ' \\\\ \\hline')

        count = count + 1
    print('')

def save_plot(plot, title):
        plot.savefig('./plots/' + title.replace(' ', '_').replace('.', 'pt').lower() + '.png')

def get_main_dfs(difficulty):
    q_path = OPTIMAL_Q_EASY if difficulty == 'easy' else OPTIMAL_Q_HARD
    value_path = difficulty.title() + ' Value.csv'
    policy_path = difficulty.title() + ' Policy.csv'

    q_df = get_df(q_path)
    value_df = get_df(value_path)
    policy_df = get_df(policy_path)

    return (q_df, value_df, policy_df)

def plot_convergence(difficulty):
    easy = difficulty == 'easy'
    q_df, value_df, policy_df = get_main_dfs(difficulty)

    plt.figure(figsize=(8,5))
    plt.title(difficulty.title() + ' Gridworld Convergence by Algorithm') 

    plt.xlabel('# Iterations')
    plt.ylabel('Convergence Delta')    
    plt.grid()
    if easy:
        plt.ylim((-0.5, 15))
        plt.xlim((0, 200))
    else:  
        plt.ylim((-0.5, 15))
        plt.xlim((0, 200))
    plt.plot(q_df['iter'].values, q_df['convergence'].values, color="b",
                     label='Q-Learning Optimal')

    plt.plot(value_df['iter'].values, value_df['convergence'].values, color="r",
                     label='Value Iteration')

    plt.plot(policy_df['iter'].values, policy_df['convergence'].values, color="g",
                     label='Policy Iteration')
    
    plt.legend(loc="best")
    save_plot(plt, difficulty + ' convergence')
    plt.close()

def plot_action(difficulty):
    easy = difficulty == 'easy'
    q_df, value_df, policy_df = get_main_dfs(difficulty)

    plt.figure(figsize=(8,5))
    plt.title(difficulty.title() + ' Gridworld Actions by Algorithm') 

    plt.xlabel('# Iterations')
    plt.ylabel('Actions')    
    plt.grid()
    #if easy:
    #    plt.ylim((-0.5, 15))
    #    plt.xlim((0, 200))
    #else:  
    #    plt.ylim((-0.5, 15))
    #    plt.xlim((0, 200))
    plt.plot(q_df['iter'].values, q_df['steps'].values, color="b",
                     label='Q-Learning Optimal')

    plt.plot(value_df['iter'].values, value_df['steps'].values, color="r",
                     label='Value Iteration')

    plt.plot(policy_df['iter'].values, policy_df['steps'].values, color="g",
                     label='Policy Iteration')
    
    plt.legend(loc="best")
    save_plot(plt, difficulty + ' steps')
    plt.close()


def plot_time(difficulty):
    easy = difficulty == 'easy'
    q_df, value_df, policy_df = get_main_dfs(difficulty)

    plt.figure(figsize=(8,5))
    plt.title(difficulty.title() + ' Gridworld Time by Algorithm') 

    plt.xlabel('# Iterations')
    plt.ylabel('Time (sec)')
    plt.grid()
    if easy:
        plt.ylim((0, 1))
        plt.xlim((0, 200))
    else:  
        plt.ylim((0, 15))
        plt.xlim((0, 500))
    
    plt.plot(q_df['iter'].values, q_df['time'].values, color="b",
                     label='Q-Learning Optimal')

    plt.plot(value_df['iter'].values, value_df['time'].values, color="r",
                     label='Value Iteration')

    plt.plot(policy_df['iter'].values, policy_df['time'].values, color="g",
                     label='Policy Iteration')
    plt.legend(loc="best")
    save_plot(plt, difficulty + ' time')
    plt.close()


def plot_reward(difficulty):
    easy = difficulty == 'easy'
    q_df, value_df, policy_df = get_main_dfs(difficulty)

    plt.figure(figsize=(8,5))
    plt.title(difficulty.title() + ' Gridworld Reward by Algorithm') 

    plt.xlabel('# Iterations')
    plt.ylabel('Reward')
    plt.grid()
    if easy:
        plt.ylim((-100, 100))
        plt.xlim((0, 500))
    else:  
        plt.ylim((-100, 70))
        plt.xlim((0, 1000))
    
    plt.plot(q_df['iter'].values, q_df['reward'].values, color="b",
                     label='Q-Learning Optimal')

    plt.plot(value_df['iter'].values, value_df['reward'].values, color="r",
                     label='Value Iteration')

    plt.plot(policy_df['iter'].values, policy_df['reward'].values, color="g",
                     label='Policy Iteration')
    plt.legend(loc="best")
    save_plot(plt, difficulty + ' reward')
    plt.close()

def plot_cumreward(difficulty):
    easy = difficulty == 'easy'
    q_df, value_df, policy_df = get_main_dfs(difficulty)
    
    q_df['cumsum'] = np.cumsum(q_df['reward'])
    value_df['cumsum'] = np.cumsum(value_df['reward'])
    policy_df['cumsum'] = np.cumsum(policy_df['reward'])

    plt.figure(figsize=(8,5))
    plt.title(difficulty.title() + ' Gridworld Cumulative Reward by Algorithm') 

    plt.xlabel('# Iterations')
    plt.ylabel('Cumulative Reward')
    plt.grid()
    #if easy:
    #    plt.ylim((-100, 70))
    #    plt.xlim((0, 500))
    #else:  
    #    plt.ylim((-100, 70))
    #    plt.xlim((0, 1000))
    
    plt.plot(q_df['iter'].values, q_df['cumsum'].values, color="b",
                     label='Q-Learning Optimal')

    plt.plot(value_df['iter'].values, value_df['cumsum'].values, color="r",
                     label='Value Iteration')

    plt.plot(policy_df['iter'].values, policy_df['cumsum'].values, color="g",
                     label='Policy Iteration')
    plt.legend(loc="best")
    save_plot(plt, difficulty + ' CumulativeReward')
    plt.close()

def plot_individual_results(policy):
    q_dfe, value_dfe, policy_dfe = get_main_dfs('easy')
    q_dfh, value_dfh, policy_dfh = get_main_dfs('hard')
    
    if policy == 'Value':
        dbe = value_dfe
        dbh = value_dfh
    if policy == 'Policy':
        dbe = policy_dfe
        dbh = policy_dfh
    if policy == 'Q-Learning':
        dbe = q_dfe
        dbh = q_dfh

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(15, 4))

    ax1.plot(dbe['iter'].values, dbe['reward'].values, color="b", label='Easy')
    ax1.plot(dbh['iter'].values, dbh['reward'].values, color="r", label='Hard')
    if policy == 'Q-Learning':
        ax1.axis(xmin=-5, xmax=1000) #, ymin=0, ymax=100)
    else:
        ax1.axis(xmin=0, xmax=100) #, ymin=0, ymax=100)
    ax1.grid(color='grey', linestyle='dotted')
    ax1.legend(loc='best', prop={'size': 8})
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Reward')
    ax1.set_title('Rewards', fontsize=8)

    ax2.plot(dbe['iter'].values, dbe['convergence'].values, color="b", label='Easy')
    ax2.plot(dbh['iter'].values, dbh['convergence'].values, color="r", label='Hard')
    if policy == 'Q-Learning':
        ax2.axis(xmin=0, xmax=500, ymin=0, ymax=20)
    else:
        ax2.axis(xmin=-5, xmax=100, ymin=-5)#, ymax=20)
    ax2.grid(color='grey', linestyle='dotted')
    ax2.legend(loc='best', prop={'size': 8})
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Convergence')
    ax2.set_title('Convergence', fontsize=8)

    ax3.plot(dbe['iter'].values, dbe['steps'].values, color="b", label='Easy')
    ax3.plot(dbh['iter'].values, dbh['steps'].values, color="r", label='Hard')
    if policy == 'Q-Learning':
        ax3.axis(xmin=0, xmax=1000, ymin=0, ymax=300)
    else:
        ax3.axis(xmin=-5, xmax=100)#, ymin=-1, ymax=100)
    ax3.grid(color='grey', linestyle='dotted')
    ax3.legend(loc='best', prop={'size': 8})
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Actions')
    ax3.set_title('Actions', fontsize=8)

    ax4.plot(dbe['iter'].values, dbe['time'].values, color="b", label='Easy')
    ax4.plot(dbh['iter'].values, dbh['time'].values, color="r", label='Hard')
    if policy == 'Q-Learning':
        ax4.axis(xmin=0, xmax=200, ymin=0, ymax=0.5)
    else:
        ax4.axis(xmin=-1, xmax=100, ymin=-1, ymax=2)
    ax4.grid(color='grey', linestyle='dotted')
    ax4.legend(loc='best', prop={'size': 8})
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Time (sec)')
    ax4.set_title('Time', fontsize=8)
    fig.tight_layout()
    for ax in fig.axes:
        ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        for item in ax_items + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(8)
    plt.subplots_adjust(wspace=0.3)
    save_plot(plt, policy + ' individual')
    plt.close()

def plot_comp_conv(difficulty):
    easy = difficulty == 'easy'
    db = get_compdf(difficulty)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

    filter = ((db['eps']==0.1) & (db['qinit']==0))
    df = db[filter]
    lrg = df.groupby('lr')
    for key, g in lrg:
        g.plot(x='iter', y='convergence', ax=ax1, label='LR %s' %key, legend=True)
    if easy:
        ax1.axis(xmin=0, xmax=500)
    else:  
        ax1.axis(xmin=0, xmax=1000)
    ax1.grid(color='grey', linestyle='dotted')
    ax1.legend(loc='best', prop={'size': 8})
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Convergence')
    ax1.set_title('Learning Rate', fontsize=8)
    
    filter = ((db['lr']==0.1) & (db['qinit']==0))
    df = db[filter]
    lrg = df.groupby('eps')
    for key, g in lrg:
        g.plot(x='iter', y='convergence', ax=ax2, label='Epsilon %s' %key, legend=True)
    if easy:
        ax2.axis(xmin=0, xmax=500, ymin=0, ymax=15)
        #ax2.xlim((0, 500))
    else:  
        ax2.axis(xmin=0, xmax=1000, ymin=0, ymax=15)
        #ax2.xlim((0, 1000))
    ax2.grid(color='grey', linestyle='dotted')
    ax2.legend(loc='best', prop={'size': 8})
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Convergence')
    ax2.set_title('Epsilon', fontsize=8)
    
    filter = ((db['lr']==0.1) & (db['eps']==0.1))
    df = db[filter]
    lrg = df.groupby('qinit')
    for key, g in lrg:
        g.plot(x='iter', y='convergence', ax=ax3, label='Q-Initial %s' %key, legend=True)
    if easy:
        ax3.axis(xmin=0, xmax=500, ymin=0, ymax=15)
        #ax3.xlim((0, 500))
    else:  
        ax3.axis(xmin=0, xmax=1000, ymin=0, ymax=15)
        #ax3.xlim((0, 1000))
    ax3.grid(color='grey', linestyle='dotted')
    ax3.legend(loc='best', prop={'size': 8})
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Convergence')
    ax3.set_title('Q-Initial', fontsize=8)
    fig.suptitle('Q-Learning Convergence')
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    save_plot(plt, difficulty + ' CompConvergence')
    plt.close()

def plot_comp_reward(difficulty):
    easy = difficulty == 'easy'
    db = get_compdf(difficulty)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

    filter = ((db['eps']==0.1) & (db['qinit']==0))
    df = db[filter]
    lrg = df.groupby('lr')
    for key, g in lrg:
        g.plot(x='iter', y='reward', ax=ax1, label='LR %s' %key, legend=True)
    if easy:
        ax1.axis(xmin=0, xmax=500)
    else:  
        ax1.axis(xmin=0, xmax=1000)
    ax1.grid(color='grey', linestyle='dotted')
    ax1.legend(loc='best', prop={'size': 8})
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Rewards')
    ax1.set_title('Learning Rate', fontsize=8)
    
    filter = ((db['lr']==0.1) & (db['qinit']==0))
    df = db[filter]
    lrg = df.groupby('eps')
    for key, g in lrg:
        g.plot(x='iter', y='reward', ax=ax2, label='Epsilon %s' %key, legend=True)
    if easy:
        ax2.axis(xmin=0, xmax=500, ymin=-300, ymax=100)
        #ax2.xlim((0, 500))
    else:  
        ax2.axis(xmin=0, xmax=1000, ymin=-300, ymax=100)
        #ax2.xlim((0, 1000))
    ax2.grid(color='grey', linestyle='dotted')
    ax2.legend(loc='best', prop={'size': 8})
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Rewards')
    ax2.set_title('Epsilon', fontsize=8)
    
    filter = ((db['lr']==0.1) & (db['eps']==0.1))
    df = db[filter]
    lrg = df.groupby('qinit')
    for key, g in lrg:
        g.plot(x='iter', y='reward', ax=ax3, label='Q-Initial %s' %key, legend=True)
    if easy:
        ax3.axis(xmin=0, xmax=500, ymin=-300, ymax=100)
        #ax3.xlim((0, 500))
    else:  
        ax3.axis(xmin=0, xmax=1000, ymin=-300, ymax=100)
        #ax3.xlim((0, 1000))
    ax3.grid(color='grey', linestyle='dotted')
    ax3.legend(loc='best', prop={'size': 8})
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Rewards')
    ax3.set_title('Q-Initial', fontsize=8)
    fig.suptitle('Q-Learning Rewards')
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    save_plot(plt, difficulty + ' CompRewards')
    plt.close()

def plot_comp_dis(difficulty):
    if difficulty == 'easy':
        q_path = 'Discount ' + OPTIMAL_Q_EASY 
    else:
        q_path = 'Discount ' + OPTIMAL_Q_HARD
    value_path = 'Discount ' + difficulty.title() + ' Value.csv'
    policy_path = 'Discount ' + difficulty.title() + ' Policy.csv'

    q_df = get_df(q_path)
    value_df = get_df(value_path)
    policy_df = get_df(policy_path)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

    lrg = q_df.groupby('discount')
    for key, g in lrg:
        g.plot(x='iter', y='reward', ax=ax1, label='Discount %s' %key, legend=True)
    #if easy:
    #    ax1.axis(xmin=0, xmax=500)
    #else:  
    #    ax1.axis(xmin=0, xmax=1000)
    ax1.grid(color='grey', linestyle='dotted')
    ax1.legend(loc='best', prop={'size': 8})
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Rewards')
    ax1.set_title('Q-Learning', fontsize=8)
    
    lrg = value_df.groupby('discount')
    for key, g in lrg:
        g.plot(x='iter', y='reward', ax=ax2, label='Discount %s' %key, legend=True)
    #if easy:
    #    ax2.axis(xmin=0, xmax=500, ymin=-300, ymax=100)
        #ax2.xlim((0, 500))
    #else:  
    #    ax2.axis(xmin=0, xmax=1000, ymin=-300, ymax=100)
        #ax2.xlim((0, 1000))
    ax2.grid(color='grey', linestyle='dotted')
    ax2.legend(loc='best', prop={'size': 8})
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Rewards')
    ax2.set_title('Value Iteration', fontsize=8)
    
    lrg = policy_df.groupby('discount')
    for key, g in lrg:
        g.plot(x='iter', y='reward', ax=ax3, label='Discount %s' %key, legend=True)
    #if easy:
    #    ax3.axis(xmin=0, xmax=500, ymin=-300, ymax=100)
        #ax3.xlim((0, 500))
    #else:  
    #    ax3.axis(xmin=0, xmax=1000, ymin=-300, ymax=100)
        #ax3.xlim((0, 1000))
    ax3.grid(color='grey', linestyle='dotted')
    ax3.legend(loc='best', prop={'size': 8})
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Rewards')
    ax3.set_title('Policy Iteration', fontsize=8)
    fig.suptitle('Discount Factor')
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    save_plot(plt, difficulty + ' CompDiscount')
    plt.close()
#find_best_q_learning('easy')
# print(' ')
#find_best_q_learning('hard')

# print_value_results('easy', 'Value')
# print_value_results('easy', 'Policy')

# print_value_results('hard', 'Value')
# print_value_results('hard', 'Policy')

plot_individual_results('Value')
plot_individual_results('Policy')
plot_individual_results('Q-Learning')

plot_convergence('easy')
plot_convergence('hard')

plot_time('easy')
plot_time('hard')

plot_reward('easy')
plot_reward('hard')

plot_comp_conv('easy')
plot_comp_conv('hard')

plot_comp_reward('easy')
plot_comp_reward('hard')

plot_cumreward('easy')
plot_cumreward('hard')

plot_action('easy')
plot_action('hard')

plot_comp_dis('easy')
plot_comp_dis('hard')