import configparser

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from pickle import dump

config = configparser.ConfigParser()
config.read_file(open('config.cfg'))

######################################
###            Utilities           ###
######################################

def cfg_read_list(cfg_section, cfg_var):
    '''Reads a sequence from config and 
    converts it to a python list

    Args:
    cfg_section - the Section of the config file
                 written as [SECTION]
    cfg_var - the variable

    Returns:
        list of strings
    '''
    return config.get(section=cfg_section, option=cfg_var).split()

def cfg_read_numbers(cfg_section, cfg_var):
    '''Reads a sequence of numbers from config and 
    converts it to a python list of floats

    Args:
    cfg_section - the Section of the config file
                 written as [SECTION]
    cfg_var - the variable

    Returns:
        list of floats
    '''
    seq = config.get(section=cfg_section, option=cfg_var).split()
    return [float(n) for n in seq]

######################################
###          Configuration         ###
######################################

SEED = config.getint('GENERAL', 'RandomSeed')
cat_features = cfg_read_list('PREPROCESSING', 'cat_features')
num_features = cfg_read_list('PREPROCESSING', 'num_features')
metric = config.get('GENERAL', 'metric')

######################################
#    Pipelines for preprocessing     #
######################################

# Initiating Pipelines
## for categorical features
cat_pipeline = Pipeline([
    ('1hot', OneHotEncoder(handle_unknown='ignore')),
])

## for numerical features
num_pipeline = Pipeline([
    ('imputer_num', SimpleImputer(strategy='constant', fill_value=999)),
    ('std_scaler', StandardScaler())
])

# Initializing the preprocessor
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

######################################
#  Pipeline for logistic regression  #
######################################

pipe_logreg = Pipeline([
    ('preprocessor', preprocessor),
    ('logreg', LogisticRegression(max_iter=1000))
])

# safer to create the list first and 
# then pass it as an argument:
### Cs = [float(C) for C in cfg_read_list('LOGREG', 'C')]

param_logreg = {'logreg__penalty': cfg_read_list('LOGREG', 'penalty'),
                'logreg__C': cfg_read_numbers('LOGREG', 'C'),
                'logreg__solver': cfg_read_list('LOGREG', 'solvers'),
                #'logreg__random_state': SEED
               }
if config.getboolean('LOGREG', 'random_search'):
    paropt_logreg = RandomizedSearchCV(pipe_logreg, 
                           param_distributions=param_logreg, 
                           cv=config.getint('GENERAL', 'cv_folds'), 
                           scoring=metric, 
                           verbose=config.getint('GENERAL', 'sklearn_verbosity'), 
                           random_state=SEED,
                           n_jobs=-1)
else: 
    paropt_logreg = GridSearchCV(pipe_logreg, 
                           param_grid=param_logreg, 
                           cv=config.getint('GENERAL', 'cv_folds'), 
                           scoring=metric, 
                           verbose=config.getint('GENERAL', 'sklearn_verbosity'), 
                           n_jobs=-1)

### Saving the pipeline ### 
fname = config.get('LOGREG', 'name')
dump(paropt_logreg, open(f'pipelines/{fname}.pkl', 'wb'))
print('Logistic regression pipeline saved as', fname)

######################################
#         Pipeline for KNN           #
######################################

# Since KNN doesn't really *learn* anything on the train data,
# the pipeline is useful mostly because of the included 
# preprocessing. Later it is possible to include it into 
# a RandomizedSearch - for hyperparameter tuning

pipe_knn = Pipeline([
    ('preprocessor', preprocessor),
    ('knn', KNeighborsClassifier(
        n_neighbors= config.getint('KNN', 'max_neighbors'),
        metric=config.get('KNN', 'dist_metric'),
        p=config.getint('KNN', 'p'),
        ))
])

#fname = config.get('KNN', 'name')
#dump(pipe_knn, open(f'pipelines/{fname}.pkl', 'wb'))
#print('KNN pipeline saved as', fname)

######################################
#    Pipeline for RandomForest       #
######################################

pipe_forest = Pipeline([
    ('preprocessor', preprocessor),
    ('forest', RandomForestClassifier())
])

#fname = config.get('FOREST', 'name')
#dump(pipe_forest, open(f'pipelines/{fname}.pkl', 'wb'))
#print('Random forest pipeline saved as', fname)

######################################
#       Pipeline for XGBoost         #
######################################

pipe_XGB = Pipeline([
    ('preprocessor', preprocessor),
    ('xgboost', XGBClassifier(eval_metric=config.get('XGBOOST', 'eval_metric'), 
                             verbosity=config.getint('XGBOOST', 'xgb_verbosity'),
                             min_split_loss=0,
                             #max_leaves=config.getint('XGBOOST', 'max_leaves'),
                             seed=SEED))
])

#learning_rates = [float(lr) for lr in cfg_read_list('XGBOOST', 'learning_rate')]
#min_split_losses = [float(msl) for msl in cfg_read_list('XGBOOST', 'min_split_loss')]
max_lvs = [int(ml)for ml in cfg_read_numbers('XGBOOST', 'max_leaves')]
param_xgb = {'xgboost__learning_rate': cfg_read_numbers('XGBOOST', 'learning_rate'),
             'xgboost__min_split_loss': cfg_read_numbers('XGBOOST', 'min_split_loss'),
             'xgboost__subsample': cfg_read_numbers('XGBOOST', 'subsample'),
             'xgboost__max_leaves': max_lvs,
             'xgboost__alpha': cfg_read_numbers('XGBOOST', 'alpha'),
             'xgboost__lambda': cfg_read_numbers('XGBOOST', 'lambda'),
             'xgboost__grow_policy': cfg_read_list('XGBOOST', 'grow_policy')
               }
paropt_XGB = RandomizedSearchCV(pipe_XGB, 
                           param_distributions=param_xgb, 
                           cv=config.getint('GENERAL', 'cv_folds'), 
                           scoring=metric, 
                           verbose=config.getint('GENERAL', 'sklearn_verbosity'),
                           random_state=SEED,
                           n_jobs=-1)

# fname = config.get('XGBOOST', 'name')
# dump(paropt_XGB, open(f'pipelines/{fname}.pkl', 'wb'))
# print('XGBoost pipeline saved as', fname)