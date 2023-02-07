'''
obtained from: https://github.com/Kaggle/learntools/blob/master/learntools/time_series/utils.py
'''

import matplotlib.pyplot as plt
import pandas as pd

def moving_avg_7_30_365_days_plots(timeseries, title, loc='upper left'):
    '''
    The [loc]ation of the legend.
    The strings 'upper left', 'upper right', 'lower left', 'lower right' place the legend at the corresponding corner of the axes/figure.
    The strings 'upper center', 'lower center', 'center left', 'center right' place the legend at the center of the corresponding edge of the axes/figure.
    The string 'center' places the legend at the center of the axes/figure.
    The string 'best' places the legend at the location, among the nine locations defined 
    '''
    plt.figure(figsize=(16,4))
    
    plt.plot(timeseries.rolling(window=7,
                        center=True,
                       ).mean(),
             label='Moving Average 7 Days');
    
    plt.plot(timeseries.rolling(window=30,
                        center=True,
                        min_periods=15,
                       ).mean(),
             label='Moving Average 30 Days');
    
    plt.plot(timeseries.rolling(window=365,
                        center=True,
                        min_periods=183,
                       ).mean(),
             label='Moving Average 365 Days');
    
    plt.title(title + " - 7/30/365-Day Moving Average");
    plt.legend(loc=loc);

# annotations: https://stackoverflow.com/a/49238256/5769929
def seasonal_plot(X, y, period, freq, ax=None):
    import seaborn as sns
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax


def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    frequencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(frequencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax

def plot_lags(x,
              y=None,
              lags=6,
              leads=None,
              nrows=1,
              lagplot_kwargs={},
              **kwargs):
    import math
    kwargs.setdefault('nrows', nrows)
    orig = leads is not None
    leads = leads or 0
    kwargs.setdefault('ncols', math.ceil((lags + orig + leads) / nrows))
    kwargs.setdefault('figsize', (kwargs['ncols'] * 2, nrows * 2 + 0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
        k -= leads + orig
        if k + 1 <= lags:
            ax = lagplot(x, y, shift=k + 1, ax=ax, **lagplot_kwargs)
            title = f"Lag {k + 1}" if k + 1 >= 0 else f"Lead {-k - 1}"
            ax.set_title(title, fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis('off')
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig

def lagplot(x, y=None, shift=1, standardize=False, ax=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    import seaborn as sns
    x_ = x.shift(shift)
    if standardize:
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y
    else:
        y_ = x
    corr = y_.corr(x_)
    if ax is None:
        fig, ax = plt.subplots()
    scatter_kws = dict(
        alpha=0.75,
        s=3,
    )
    line_kws = dict(color='C3', )
    ax = sns.regplot(x=x_,
                     y=y_,
                     scatter_kws=scatter_kws,
                     line_kws=line_kws,
                     lowess=True,
                     ax=ax,
                     **kwargs)
    at = AnchoredText(
        f"{corr:.2f}",
        prop=dict(size="large"),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    title = f"Lag {shift}" if shift > 0 else f"Lead {shift}"
    ax.set(title=f"Lag {shift}", xlabel=x_.name, ylabel=y_.name)
    return ax

'''
obtained from: https://gist.github.com/javiferfer/ba280cae79a86d6994ac8351df1f3756
'''
def analyze_stationarity(timeseries, title):
    fig, ax = plt.subplots(3, 1, figsize=(16, 10))

    rolmean = pd.Series(timeseries).rolling(window=7, center=True).mean()
    rolstd = pd.Series(timeseries).rolling(window=7, center=True).std()
    ax[0].plot(timeseries, label= title, alpha=0.5)
    ax[0].plot(rolmean, label='Moving Average');
    ax[0].plot(rolstd, label='Moving Std (x10)', alpha=0.7);
    ax[0].set_title('7-Day Window')
    ax[0].legend()
    
    rolmean = pd.Series(timeseries).rolling(window=30, center=True, min_periods=15).mean() 
    rolstd = pd.Series(timeseries).rolling(window=30, center=True, min_periods=15).std()
    ax[1].plot(timeseries, label= title, alpha=0.5)
    ax[1].plot(rolmean, label='Moving Average');
    ax[1].plot(rolstd, label='Moving Std (x10)');
    ax[1].set_title('30-Day Window')
    ax[1].legend()
    
    rolmean = pd.Series(timeseries).rolling(window=365, center=True, min_periods=183).mean() 
    rolstd = pd.Series(timeseries).rolling(window=365, center=True, min_periods=183).std()
    ax[2].plot(timeseries, label= title, alpha=0.5)
    ax[2].plot(rolmean, label='Moving Average');
    ax[2].plot(rolstd, label='Moving Std (x10)');
    ax[2].set_title('365-Day Window')
    ax[2].legend()

def moving_avg_7_30_365_days_plots_alternative(timeseries, title):
    fig, ax = plt.subplots(figsize=(16, 4))

    rolmean_7 = pd.Series(timeseries).rolling(window=7, center=True).mean()
    rolmean_30 = pd.Series(timeseries).rolling(window=30, center=True, min_periods=15).mean()
    rolmean_365 = pd.Series(timeseries).rolling(window=365, center=True, min_periods=183).mean()
    ax.plot(timeseries, label= title, alpha=0.7)
    ax.plot(rolmean_7, label='Moving Average 7 Days');
    ax.plot(rolmean_30, label='Moving Average 30 Days');
    ax.plot(rolmean_365, label='Moving Average 365 Days');
    ax.set_title(title + ' - 7/30/365-Day Window')
    ax.legend()