"""
Predicts job satisfaction based on models trained by the Stack Overflow Developer Survey.
"""
import xgboost as xgb

FEATS_2015 = {
    'Compensation': 'salary',
    'Purchasing Power_I have no say in purchasing what I need or want at work': 'choose_equip',
    'Remote Status_Never': 'remote',
    'Changed Jobs in last 12 Months': 'curr_job_less_than_year',
}

FEATS_2016 = {
    'agree_loveboss': 'like_boss',
    'agree_tech': 'job_technologies',
    'open_to_new_job_I am actively looking for a new job': 'look_postings_frequent',
    'interview_likelihood': 'interview',
}

FEATS_2017 = {
    'CareerSatisfaction': 'like_developer',
    'HoursPerWeek': 'hours_per_week',
    'Overpaid': 'overpaid',
    'LastNewJob_Less than a year ago': 'curr_job_less_than_year',
    'InfluenceWorkstation': 'choose_equip',
    'Salary': 'salary',
}

XGB_2015 = xgb.Booster(model_file='output2015.model')
XGB_2016 = xgb.Booster(model_file='output2016.model')
XGB_2017 = xgb.Booster(model_file='output2017.model')

MODELS = (
    (XGB_2015, FEATS_2015),
    (XGB_2016, FEATS_2016),
    (XGB_2017, FEATS_2017),
)

NUM_MODELS = len(MODELS)
REQUIRED_KEYS = set(feat for model in MODELS for feat in model[1].values())

def clip(num, low, high):
    """
    Replicates np.clip.
    """
    if num < low:
        return low

    if num > high:
        return high

    return num

def predict_year(user_in, year):
    """
    Get the prediction for a specific year's model.
    """
    feats = MODELS[year][1].keys()
    model_in = [float(user_in[MODELS[year][1][feat]]) for feat in feats]
    model_in = xgb.DMatrix(model_in, feature_names=feats)

    return MODELS[year][0].copy().predict(model_in)

def predict(user_in):
    """
    Predict job satisfaction using the data provided.
    If user_in does not contain all REQUIRED_KEYS, returns -1.
    """
    if not all(key in user_in for key in REQUIRED_KEYS):
        return -1

    out = sum(predict_year(user_in, i) for i in xrange(NUM_MODELS)) / NUM_MODELS
    return clip(out, 0, 10)
