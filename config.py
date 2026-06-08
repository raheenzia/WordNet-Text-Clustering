EXPERIMENT_CONFIGS = {
    'Baseline': {
        'use_pos': False,
        'use_synonyms': False,
        'hypernym_depth': 0
    },
    'PoS_Only': {
        'use_pos': True,
        'use_synonyms': False,
        'hypernym_depth': 0
    },
    'Syns': {
        'use_pos': True,
        'use_synonyms': True,
        'hypernym_depth': 0
    },
    'Hyper_5': {
        'use_pos': True,
        'use_synonyms': True,
        'hypernym_depth': 5
    },
    'Hyper_All': {
        'use_pos': True,
        'use_synonyms': True,
        'hypernym_depth': -1 
    }
}

CLUSTER_SIZES = [15, 20 , 25]
NUM_RUNS = 3  
RANDOM_SEED = 42
PRUNING_THRESHOLD = 200