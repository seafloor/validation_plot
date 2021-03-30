import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from skmisc.loess import loess
from .predict import yhat
from .format import format_train
from tqdm.notebook import tqdm
import pandas as pd
import os


def fit_calibrator(X_train, y_train, X_test, cal='lr', seed=12345):
    """
    Parameters
    ----------
    X_train: ndarray
        test-fold predictions from base classifier from CV
    y_train: ndarray
        labels for test-folds
    X_test: ndarray
        separate test set predictions from base classifier after refit
    cal: str, default 'lr'
        estimator to use as calibrator
    seed: int, default 12345
        passed for instatiation of calibrator
    Returns
    -------
    yhat: ndarray
        calibrated predictions on the test set
    """
    if cal == 'lr':
        est = LogisticRegression(penalty='none', solver='newton-cg', random_state=seed)
    else:
        raise ValueError('Only logistic calibration supported')
    est.fit(format_train(X_train), y_train)

    return yhat(est, format_train(X_test))


def elkan_transform(y_pred, base_rate, new_rate):
    # getting base_rate and new_rate
    base_rate = base_rate if isinstance(base_rate, float) else np.mean(base_rate)
    new_rate = new_rate if isinstance(new_rate, float) else np.mean(new_rate)

    # Elkan's formula
    numerator = y_pred - (y_pred * base_rate)
    denominator = base_rate - (y_pred * base_rate) + (new_rate * y_pred) - (base_rate * new_rate)
    frac = numerator / denominator

    return new_rate * frac


def hastie_tibshirani_transform(y_pred, base_log_odds, new_log_odds):
    # getting base_log_odds and new_log_odds
    # pass log odds as float or numpy array to get log odds from
    base_log_odds = base_log_odds if isinstance(base_log_odds, float) else log_odds(base_log_odds)
    new_log_odds = new_log_odds if isinstance(new_log_odds, float) else log_odds(new_log_odds)

    return y_pred - base_log_odds + new_log_odds


def log_odds(outcome):
    p = np.mean(outcome)

    return np.log(p / (1 - p))


def summarise_calibration(y_true, y_pred):
    return {'Brier score': brier_score_loss(y_true, y_pred),
            'O/E': np.sum(y_true) / np.sum(y_pred),
            'AUC': roc_auc_score(y_true, y_pred)}


def plot_calibration(y_true, y_pred, label='classifier', quantiles=10, add_distribution='None', lim_frug=True,
                     add_groupings=True, axislim=(0, 1), fig=None, ax=None, annot=False, annot_x=0.7, annot_y=0.3,
                     annot_pad=12, add_ci=True, loess_label='Loess smoother', tick_freq=10):
    fig, ax, ax1 = setup_axes(fig, ax, add_distribution)
    df = pd.DataFrame({'y_true': y_true.copy(), 'y_pred': y_pred.copy()})

    if add_groupings:
        df_aggregated = group_observations(df['y_true'], df['y_pred'], quantiles=quantiles)

    pred, conf, smlowess, ll, ul = fit_lowess(df['y_pred'], df['y_true'])
    df_plot = pd.DataFrame({'x': df['y_pred'].values, 'y': smlowess, 'y1': ll, 'y2': ul,
                          'y_true': df['y_true'].values}).sort_values('x', ascending=True)

    if smlowess is not None:
        ax.plot(df_plot.x, df_plot.y, 'red', linewidth=1, label=loess_label)
    if all([add_ci, ll is not None, ul is not None]):
        ax.fill_between(x=df_plot.x,
                        y1=df_plot.y1,
                        y2=df_plot.y2,
                        color='lightgrey', alpha=0.6)

    if add_groupings:
        ax.errorbar(df_aggregated['y_pred_mean'], df_aggregated['y_true_mean'], yerr=1.96 * df_aggregated['y_true_sem'],
                    fmt='', marker='o', linestyle='', label='Grouped Obs.')

    ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    ax.set_ylim(axislim[0], axislim[1])
    ax.set_xlim(axislim[0], axislim[1])
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed frequency')
    ax.set_title(label) if ax1 is None else ax1.set_title(label)
    ax.legend(frameon=False, loc='upper left')
    tick_frequency = axislim[1]/tick_freq
    ax.set_xticks(np.arange(axislim[0], axislim[1]+tick_frequency, tick_frequency))
    ax.set_yticks(np.arange(axislim[0], axislim[1]+tick_frequency, tick_frequency))
    sns.despine()

    if annot is not False:
        assert isinstance(annot, (str, list, tuple)), 'supply string or iterable'
        if isinstance(annot, str):
            annot = [annot]
        annotation_dict = summarise_calibration(y_true, y_pred)
        for i, annotation in enumerate(annot):
            value = annotation_dict.get(annotation, None)
            if value is not None:
                ax.text(x=annot_x, y=annot_y-(i / annot_pad), s='{}: {:.2g}'.format(annotation, value))

    if add_distribution != 'None':
        assert ax1 is not None, 'Must have second axis for rug plot'
        ax = plot_calibration_distribution(add_distribution, ax, ax1, df_plot, y_true, axislim, lim_frug, tick_frequency)

    return fig, ax


def plot_calibration_grid(y_true, y_pred, label, nrows, ncols, figsize=(16, 8), fig=None, ax=None, progress_bar=False,
                          tick_freq=10, add_groupings=True, add_ci=True, add_distributions='kde'):
    if ax is None:
        fig, ax = setup_grid_axes(nrows, ncols, figsize)

    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'label': label})
    df_grouped = df.groupby('label')
    for i, (_, g) in tqdm(enumerate(df_grouped), total=len(df_grouped), desc='creating subplots',
                          disable=not progress_bar):
        plot_calibration(g.y_true, g.y_pred, g.label.unique()[0], add_groupings=add_groupings,
                         add_distribution=add_distributions, axislim=(0, 1), fig=fig, ax=ax[i], tick_freq=tick_freq,
                         add_ci=add_ci)

    return fig, ax


def setup_axes(fig, ax, add_distribution='None'):
    if fig is None:
        ysize = 8 if add_distribution == 'None' else 10
        fig = plt.figure(figsize=(10, ysize))

    if ax is None:
        gridsize = (6, 1)
        if add_distribution != 'None':
            ax = plt.subplot2grid(gridsize, (1, 0), rowspan=5)
            ax1 = plt.subplot2grid(gridsize, (0, 0), rowspan=1)
        else:
            ax = plt.subplot2grid(gridsize, (0, 0), rowspan=6)
            ax1 = None
    else:
        if isinstance(ax, (tuple, list)):
            assert add_distribution != 'None',\
                'You must pass a single axes object to ax if not plotting a marginal distribution'
            ax, ax1 = ax
        else:
            assert add_distribution == 'None', 'You must pass a list of axes to ax to plot a marginal distribution'
            ax1 = None

    return fig, ax, ax1


def adjust_for_row(i, row):
    return int(((np.floor(i/row))*row)+i)


def setup_grid_axes(nrows, ncols, figsize, marginal_height=1/7):
    heights = [marginal_height, 1] * nrows
    fig, ax = plt.subplots(figsize=figsize, nrows=nrows*2, ncols=ncols, constrained_layout=True,
                           gridspec_kw={'height_ratios': heights})
    ax = ax.flatten()

    axis_groups = [[adjust_for_row(i, ncols), adjust_for_row(i, ncols) + ncols] for i, cax in enumerate(ax)]
    axis_groups = [axis_groups[i] for i in range(nrows*ncols)]
    grid_axes = [(ax[j], ax[i]) for i, j in axis_groups]

    return fig, grid_axes


def group_observations(y_true, y_pred, quantiles=10):
    df = pd.DataFrame({'y_true': y_true.copy(), 'y_pred': y_pred.copy()})
    df['decile'] = pd.qcut(df['y_pred'], q=quantiles)
    df_aggregated = df.groupby('decile').agg(['mean', 'sem', 'count']).loc[:, ['y_pred', 'y_true']]
    df_aggregated.columns = ['_'.join(col).strip() for col in df_aggregated.columns.values]

    return df_aggregated


def fit_lowess(y_pred, y_true):
    l = loess(y_pred, y_true)
    pred, conf, smlowess, ll, ul = None, None, None, None, None

    try:
        l.fit()
        pred = l.predict(y_pred, stderror=True)
        conf = pred.confidence()
        smlowess = pred.values
        ll = conf.lower
        ul = conf.upper
    except ValueError as e:
        print(e)

    return pred, conf, smlowess, ll, ul


def plot_calibration_distribution(add_distribution, ax, ax1, df, y_true, axislim, limit_xaxis, tick_frequency, a=0.25):
    if add_distribution == 'rug':

        ax1 = mirrored_histogram_rug(df, y_true=y_true, ax=ax1, axislim=axislim, limit_xaxis=limit_xaxis,
                                     tick_frequency=tick_frequency)
    elif add_distribution == 'hist':
        ax1.hist(df.query('y_true == "1"').x, alpha=a, label='1')
        ax1.hist(df.query('y_true == "0"').x, alpha=a, label='0')
        ax1.legend(frameon=False)
    elif add_distribution == 'kde':
        sns.kdeplot(df.query('y_true == "1"').x, alpha=a, ax=ax1, shade=True, label='1')
        sns.kdeplot(df.query('y_true == "0"').x, alpha=a, ax=ax1, shade=True, label='0')
        ax1.legend(frameon=False)
    else:
        assert add_distribution == 'None'

    ax1.axis('off')
    ax = [ax, ax1]

    return ax


def mirrored_histogram_rug(df, y_true='y_true', ax=None, axislim=(0, 1), limit_xaxis=True, tick_frequency=None):
    # setup parameter for rug plot
    tick_frequency = tick_frequency if tick_frequency is not None else axislim[1]/10
    xmin = df.x.min() if limit_xaxis else 0
    xmax = df.x.max() if limit_xaxis else 1

    # create histogram counts by outcome status
    df['bins'] = round(df.x, 2)
    g = df.groupby(['bins', 'y_true']).size().reset_index()
    g.columns = ['bins', 'y_true', 'y_count']
    g['y_scale_count'] = g['y_count'] / g['y_count'].max()

    # plot histogram by outcome status
    ax.bar(x=g.loc[g['y_true'] == 1, 'bins'],
           height=g.loc[g['y_true'] == 1, 'y_scale_count'],
           width=0.007, color='grey')
    ax.bar(x=g.loc[g['y_true'] == 0, 'bins'], height=-g.loc[g['y_true'] == 0, 'y_scale_count'], width=0.007, color='grey')
    ax.axhline(y=0, xmin=xmin, xmax=xmax, color='grey', linestyle='-', linewidth=1)
    ax.set_xlim(axislim[0], axislim[1])
    ax.set_ylim(-1, 1)
    ax.text(x=axislim[1] - (tick_frequency / 5), y=0.15, s='1', fontsize=10, color='dimgrey')
    ax.text(x=axislim[1] - (tick_frequency / 5), y=-0.35, s='0', fontsize=10, color='dimgrey')

    return ax
