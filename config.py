args={}

args['loss'] = "cross_entropy" # mnist: "nll_loss", cifar10: "cross_entropy"
args['batch_size']=10000
args['test_batch_size']=500
args['epochs']=2  #The number of Epochs is the number of times you go through the full dataset. 
args['lr']=0.01 #Learning rate is how fast it will decend. 
args['momentum']=0.5 #SGD momentum (default: 0.5) Momentum is a moving average of our gradients (helps to keep direction).
args['seed']=1 #random seed
args['log_interval']=10
args['cuda']=False

args['ALPHA'] = 0.99 # mnist, cifar10: 0.9
args['BETA'] = 0.6 # cifar10: 0.7 mnist: 0.7
args['MIN_MATCH']=0.80
args['TIME_BUDGET']= 3 # seconds
args['MIN_SCORE_TO_FILTER']=0.05
args['batch_size'] = 128
args['PCA_n_components'] = [16, 32, 64, 128] # cifar10: [64, 128] # mnist: [16, 32, 64, 128]
args['Birch_thresholds'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
args['KMEANS_n_clusters'] = [2, 5, 7, 10]
args['SFL_strategy'] = "barinel" # tarantula, ochiai, barinel
