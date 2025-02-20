import torch
from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.optimize import minimize

"""
Code for determining the factors for metric D (performance difference between few-shot and many-shot) in the paper.
"""

def quadratic_form(x, A):
    """
    Computes f(x) = [x, 1]^T * A * [x, 1],
    where x is a 1D array of length n, and A is (n+1)x(n+1).
    """
    # Extend x by 1 to handle linear & constant terms
    x_ext = np.concatenate([x, [1.0]])
    return x_ext @ A @ x_ext


def find_min_max(A):
    """
    Finds the minimum and maximum of f(x) = x_ext^T A x_ext,
    subject to x_i in [0,1].

    Returns:
      fmin, x_at_min, fmax, x_at_max
    """
    n = A.shape[0] - 1  # because A is (n+1)x(n+1)
    
    # Initial guess (you might try multiple guesses if non-convex)
    x0 = np.full(n, 0.5)

    # Bounds: each x_i in [0,1]
    bounds = [(0, 1)] * n

    # --- 1) Minimize f(x):
    res_min = minimize(
        fun=quadratic_form,
        x0=x0,
        args=(A,),
        method='SLSQP',
        bounds=bounds
    )

    # --- 2) Maximize f(x) by minimizing -f(x):
    res_max = minimize(
        fun=lambda x: -quadratic_form(x, A),
        x0=x0,
        method='SLSQP',
        bounds=bounds
    )

    fmin = res_min.fun
    xmin = res_min.x

    fmax = -res_max.fun
    xmax = res_max.x

    return fmax, xmax

"""
{'problem idx': ww[i][0],
                             'our metric': ww[i][1], 
                             'codelen': codelen[ww[i][0]],
                             'shotlen': shotlen[ww[i][0]],
                             'difficulty': difficulty[ww[i][0]],
                             '4-shot acc': scr_final[ww[i][0], 4],
                             '8-shot acc': scr_final[ww[i][0], 8],
                             '16-shot acc': scr_final[ww[i][0], 16],
                             '32-shot acc': scr_final[ww[i][0], 32],
                             '64-shot acc': scr_final[ww[i][0], 64],
                             '128-shot acc': scr_final[ww[i][0], 128],
                             '256-shot acc': scr_final[ww[i][0], 256],
                             'num_different_answer': len(list(set([str(x) for x in data_output]))),
                             'most_common_answer_ratio': Counter([str(x) for x in data_output]).most_common(1)[0][1] / len(data_output),
                             'code': code
                             }
"""

FACTOR =  ['64-shot-acc', 'difficulty', 'most_common_answer_ratio','num_different_answer',  'shotlen', 'codelen'] # 'num_different_answer', 'most_common_answer_ratio',
def fit_quadratic(data, FACTOR):
    """
    Fits a quadratic function y = f(factor1, factor2, ..., factor6) based on input data.
    
    Args:
        data (list of dict): A list where each element is a dictionary with keys
                             'factor1', 'factor2', ..., 'factor6', and 'y'. 
                             All values are floats.
    
    Returns:
        model: The fitted model (scikit-learn pipeline).
    """
    # Extract features (factors) and target (y)
    factors = FACTOR
    X = np.array([[element[factor] for factor in factors] for element in data])
    y = np.array([element['our metric'] for element in data])
    
    # Create a pipeline with polynomial features (degree=2) and a linear regression model
    model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
    
    # Fit the model to the data
    model.fit(X, y)
    # print("model:", model)
    return model



def fit(x, y):
    """
    Fit the model y = k * log(x) + b to the data.

    Parameters:
    x (array-like): Independent variable data.
    y (array-like): Dependent variable data.

    Returns:
    k (float): Slope of the fitted line.
    b (float): Intercept of the fitted line.
    """
    # Convert inputs to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    #print("x:", x)
    # Compute the logarithm of x
    # Perform linear regression on y vs. log(x)
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # Return the coefficients k and b
    return slope, intercept

data = torch.load("problem_summary_from_analyzer_weight_v2.pt")
# v2 = original; v3 = fixed the problem of evaluation

for i in range(len(data)):
    problem = data[i]
    for j in range(2, 8):
        data[i][str(2 << j) + '-shot-acc'] = data[i][str(2 << j) + '-shot acc']
    problem['codelen'] = len(problem['code'])

for factor in FACTOR:
    if factor == '64-shot-acc': NORM_CONST = 5
    elif factor == 'most_common_answer_ratio': NORM_CONST = 0.5#1
    elif factor == "num_different_answer": NORM_CONST = 20000#1
    elif factor == 'shotlen': NORM_CONST = 128#1
    elif factor == 'codelen': 
        arr = np.array([data[i][factor] for i in range(len(data))])
        NORM_CONST = arr.max()#1
    for i in range(len(data)):
       data[i][factor] /= NORM_CONST

########### quadratic fit ############

def build_quadratic_matrix(coeffs, FACTOR, NUM, plot=True):
    """
    Build a quadratic coefficient matrix Q from a dictionary of coefficients.
    
    Parameters
    ----------
    coeffs : dict
        Dictionary where keys are either:
        - Variable name (linear term), e.g. "x"
        - Variable squared, e.g. "x^2"
        - Cross terms, e.g. "x y"
      Values are the coefficients.
      
    plot : bool
        If True, display a seaborn heatmap of the Q matrix.
        
    Returns
    -------
    Q : np.ndarray
        The symmetric matrix of quadratic coefficients.
    variables : list of str
        The list of variable names in the order used for Q.
    """
    
    # Parse keys to find all variable names
    # Keys can be:
    # - single var (linear): e.g. "64-shot acc"
    # - squared var: e.g. "64-shot acc^2"
    # - cross terms: "64-shot acc difficulty"
    
    vars_set = set()
    for k in coeffs.keys():
        if "^2" in k:
            # e.g. "x^2"
            var = k.replace("^2", "").strip()
            vars_set.add(var)
        elif " " in k:
            # Cross term: split by space
            var1, var2 = k.split(" ", 1)
            vars_set.add(var1.strip())
            vars_set.add(var2.strip())
        else:
            # Linear term
            vars_set.add(k.strip())
    
    # Convert to a sorted list (or any ordering you prefer)
    variables = sorted(vars_set)
    # print("variables:", variables)
    # Create a map from variable to index
    var_to_idx = {v: i for i, v in enumerate(variables)}
    
    # Initialize Q matrix
    n = len(variables)
    Q = np.zeros((n+1, n+1))
    
    # Fill Q with quadratic and interaction terms
    for k, val in coeffs.items():
        k = k.strip()
        if "^2" in k:
            # Diagonal term
            var = k.replace("^2", "").strip()
            i = var_to_idx[var]
            Q[i, i] = val
        elif " " in k:
            # Cross term: split and assign half to symmetric positions
            var1, var2 = k.split(" ", 1)
            var1 = var1.strip()
            var2 = var2.strip()
            i = var_to_idx[var1]
            j = var_to_idx[var2]
            # Each cross term coefficient should be split in half for symmetric matrix
            Q[i, j] = val / 2
            Q[j, i] = val / 2
        else:
            # Linear terms are ignored in Q
            var = k.strip()
            i = var_to_idx[var]
            Q[n, i] = Q[i, n] = val / 2
    
    # Plot the heatmap if requested
    if plot:
        plt.figure(figsize=(8, 6))

        #name_variables = {'64-shot-acc': '64-shot Acc.', 'difficulty': 'Difficulty', 'most_common_answer_ratio': 'Most Common Answer Ratio', 'num_different_answer': "\# Different Answer", 'shotlen': 'Shot Length', 'codelen': 'Code Length'} # ['64-shot-acc', 'difficulty', 'most_common_answer_ratio','num_different_answer',  'shotlen', 'codelen']
        v = ['64-shot Acc.', 'Code Length', 'Difficulty', 'Most Common Ratio', '# Different Answer', 'Shot Length']
        #for x in variables:
        #    v[x] = name_variables[x]
        
        for i in range(64):
            x0 = np.zeros(6)
            for j in range(6):
                x0[j] = float((i >> j) % 2)
            print("x0:", x0)
            res = find_min_max(Q)
            print('res:', res)

        ax = sns.heatmap(Q, annot=True, xticklabels=v, yticklabels=v, 
                         cmap="coolwarm", center=0, fmt=".3g",annot_kws={"size": 12})
        plt.xticks(rotation=25)
        cbar = ax.collections[0].colorbar  # Get the colorbar
        cbar.ax.tick_params(labelsize=12)  # Adjust tick label font size
        ax.set_title('Quadratic Coefficients Heatmap', fontsize=17)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)

        plt.tight_layout()
        # v2 = original; v3 = fixed the problem of evaluation
        plt.savefig('pic/paramsearch-normalized/tmp_'+str(NUM)+'_v3.pdf', bbox_inches='tight')
        plt.clf()
    return Q, variables

def fit_quad(FACTOR, data, NUM):
    model = fit_quadratic(data, FACTOR)
    # Coefficients and intercept of the quadratic model
    coefficients = model.named_steps['linearregression'].coef_
    intercept = model.named_steps['linearregression'].intercept_

    #print("Coefficients:", coefficients)
    #print("Intercept:", intercept)

    poly = model.named_steps['polynomialfeatures']
    feature_names = poly.get_feature_names_out(FACTOR)
    print(feature_names)
    for x, y in zip(coefficients, feature_names):
        print(x, y)
    
    X = np.array([[element[factor] for factor in FACTOR] for element in data])
    # Get predictions
    y_pred = model.predict(X)
    for i in range(len(data)):
        data[i]['pred_our_metric'] = y_pred[i]

    data = sorted(data, key=lambda x: x['pred_our_metric'], reverse=True)
    metric = np.array([data[i]['our metric'] for i in range(300)])
    print("metric:", metric.mean())
    # print("Predicted y values:", y_pred)
    build_quadratic_matrix({feature_names[i]: coefficients[i] for i in range(len(feature_names))}, FACTOR, NUM, plot=True)
    return np.array([data[i]['problem idx'] for i in range(300)]), np.array([data[i]['code'] for i in range(300)])

##############
from copy import deepcopy

for i in [63]:
    # if i not in lst: continue
    factor = []
    for j in range(len(FACTOR)):
        if i & (1 << j):
            factor.append(FACTOR[j])
    
    if len(factor) == 0:
        continue
    print("======================")
    print("i:", i, "factor:", factor)
    selected, codes = fit_quad(factor, data, i)