from __future__ import division
import numpy as np
from scipy.optimize import fmin
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter
import seaborn as sns

from packages.miscellaneous import miscFuncs as misc

sns.set_style('ticks')
sns.set_context('talk')


def computeVarCorCovCE(
        spk_counts, count_means, trial_counts, phi,
        times=None, min_trials=6, prop_cutoff=0.5):

    """computeVarCorCovCE(spk_counts, count_means, trial_counts, phi,
                          times=None, min_trials=6, prop_cutoff=0.5)

    Compute variance, correlation and covariance of the conditional expectation
    of spike counts

    Parameters:
    -----------

    spk_counts:
      2D array of shape (n_trials, n_times); typically will be an array that
      contains individual trial spike counts from many cells and conditions;
      an intuitive arrangment is to have contiguous trials correspond to a
      particular cell/condition combination, and larger blocks correspond to
      all conditions for a given cell

    count_means:
      2D array of same shape as spk_counts; for each trial within spk_counts,
      count_means gives the mean spike count at time t for the group of trials
      to which that spike count belongs; a group of trials here is a particular
      grouping of cell and condition

    trial_counts:
      2D array of same shape as spk_counts; for each trial within spk_counts,
      trial_counts gives the number of trials at time t for the group of trials
      to which that spike count belongs; a group of trials here is a particular
      grouping of cell and condition

    phi:
      either a float (which means use same phi for all cells) or 2D array of
      same shape as spk_counts (which means use different phi for each cell)

    times:
      1D array of length ntimes, indicating the actual time of each time bin

    min_trials:
      For a condition (cell/condition grouping) to contribute to the
      grand average, it must have at least min_trials trials remaining at
      that timepoint.

    prop_cutoff:
      float, attrition rule


    Returns:
    --------
      dictionary with the relevant quantities

    """
    spk_counts = spk_counts.copy()
    count_means = count_means.copy()
    if times is not None:
        times = times.copy()

    spk_counts[trial_counts < min_trials] = np.nan
    count_means[trial_counts < min_trials] = np.nan

    n_remain = np.sum(~(np.isnan(spk_counts)), axis=0)
    p_remain = n_remain / n_remain[0]
    p_terminated = 1 - p_remain

    spk_counts = spk_counts[:, p_terminated <= prop_cutoff]
    count_means = count_means[:, p_terminated <= prop_cutoff]
    if np.ndim(phi) > 1:
        phi = phi[:, p_terminated <= prop_cutoff]
    residuals = spk_counts - count_means
    masked_residuals = np.ma.array(residuals, mask=np.isnan(residuals))
    covCE = np.ma.getdata(
        np.ma.cov(masked_residuals, rowvar=False, allow_masked=True))

    varCE = np.diag(covCE) - np.nanmean(phi * count_means, axis=0)

    corCE = covCE
    corCE[np.diag_indices_from(corCE)] = varCE
    corCE = corCE / np.sqrt(np.outer(varCE, varCE))

    g_out = {}
    g_out['phi'] = phi
    g_out['varCE'] = varCE
    g_out['corCE'] = corCE
    g_out['covCE'] = covCE

    if times is not None:
        g_out['times'] = times[p_terminated < prop_cutoff]

    return g_out


def simulateDiffusion(
        mean_slopes=[40], n_trials_per_cond=5000, trial_length=0.5,
        dt=.001, step_size=60, bin_size=60):

    lambda0 = 20
    v = 21.8

    n_trials = len(mean_slopes) * n_trials_per_cond
    n_times = trial_length / dt

    lambdas = []
    for slope in mean_slopes:
        init = np.ones([n_trials_per_cond, 1]) * lambda0
        noise = np.random.randn(
            n_trials_per_cond, n_times - 1) * v * np.sqrt(dt)
        noise += slope * dt
        lambdas.append(np.cumsum(np.c_[init, noise], axis=1))
    lambdas = np.concatenate(lambdas)
    lambdas[lambdas < 0] = 0

    # generate spikes with p(spike) = lambda * dt
    spk_bins = np.random.rand(
        n_trials, np.int16(trial_length / dt)) < lambdas * dt

    g_out = {}
    g_out['lambdas'] = lambdas
    g_out['spk_bins'] = spk_bins
    g_out['spk_bin_times'] = np.arange(0, trial_length / dt)
    g_out['labels'] = mean_slopes

    time_windows_begin = np.arange(0, n_times - bin_size + 1,
                                   step_size)
    time_windows_end = time_windows_begin + bin_size
    time_windows = np.c_[time_windows_begin, time_windows_end]
    spk_bin_times = np.arange(0, n_times)

    integrated_lambdas = np.zeros([lambdas.shape[0], len(time_windows_begin)])
    spk_counts = np.empty_like(integrated_lambdas)
    count_means = np.empty_like(integrated_lambdas)
    for i in xrange(len(time_windows)):
        integrated_lambdas[:, i] = np.sum(lambdas[:, np.logical_and(
            spk_bin_times >= time_windows[i][0],
            spk_bin_times < time_windows[i][1])], axis=1) * dt
        spk_counts[:, i] = np.sum(spk_bins[:, np.logical_and(
            spk_bin_times >= time_windows[i][0],
            spk_bin_times < time_windows[i][1])], axis=1)

    trial_ctr = 0
    for i in mean_slopes:
        count_means[trial_ctr:trial_ctr + n_trials_per_cond] = np.mean(
            spk_counts[trial_ctr:trial_ctr + n_trials_per_cond], axis=0)
        trial_ctr += n_trials_per_cond

    g_out['ilambdas'] = integrated_lambdas
    g_out['spk_counts'] = spk_counts
    g_out['count_means'] = count_means
    g_out['spk_counts_times'] = np.mean(time_windows, axis=1)
    g_out['trial_counts'] = np.empty_like(count_means)
    g_out['trial_counts'][:] = n_trials_per_cond
    g_out['phi'] = np.ones(count_means.shape) - 0.1
    return g_out


def simulateFindingBestCorCE(
        mean_slopes=[-40, -20, -10, -5, 0, 5, 10, 20, 40, 80, 160],
        n_sims_per_slope=25,
        n_trials_per_sim=20000,
        n_bins=6):
    phis = np.zeros([len(mean_slopes), n_sims_per_slope])
    corCEs = np.zeros([len(mean_slopes), n_sims_per_slope,
                       n_bins, n_bins])
    for i, mean_slope in enumerate(mean_slopes):
        print mean_slope
        for j in range(n_sims_per_slope):
            g = simulateDiffusion([mean_slope],
                                  n_trials_per_cond=n_trials_per_sim,
                                  step_size=60, bin_size=60)
            phi = findBestPhi(g['spk_counts'][:, 0:n_bins],
                              g['count_means'][:, 0:n_bins],
                              g['trial_counts'][:, 0:n_bins])
            # store phi
            phis[i, j] = phi

            # get the best fit corCE values
            g_var = computeVarCorCovCE(
                g['spk_counts'][:, 0:n_bins],
                g['count_means'][:, 0:n_bins],
                g['trial_counts'][:, 0:n_bins],
                phi)
            corCEs[i, j, :, :] = g_var['corCE']
    corCETheory = corCEpartialSums(n_bins)

    corCEsMean = np.mean(corCEs, axis=1)
    corCEsStd = np.std(corCEs, axis=1)
    fig = plt.figure()
    for i, mean_slope in enumerate(mean_slopes):
        ax = fig.add_subplot(3, 4, i + 1)
        x = np.arange(2, n_bins + 1)

        ax.scatter(
            x, np.diagonal(corCEsMean[i, :, :], 1),
            color='k', s=50, facecolors='none')
        ax.errorbar(x, np.diagonal(corCEsMean[i, :, :], 1),
                    yerr=np.diagonal(corCEsStd[i, :, :], 1),
                    linestyle='none', capsize=0, color='k', elinewidth=1)

        ax.scatter(x, corCEsMean[i, 0, 1:], color='k', s=50)
        ax.errorbar(x, corCEsMean[i, 0, 1:],
                    yerr=corCEsStd[i, 0, 1:], linestyle='none', capsize=0,
                    color='k', elinewidth=1)

        ax.add_line(
            plt.Line2D(x, np.diagonal(corCETheory, 1),
                       color='k', linestyle='--', linewidth=1))
        ax.add_line(plt.Line2D(x, corCETheory[0, 1:], color='k', linewidth=1))

        ax.grid('on')
        ax.set_xlim(1.75, n_bins + .25)
        ax.set_ylim(0.1, 1)
        majorLocator = MultipleLocator(1)
        ax.xaxis.set_major_locator(majorLocator)

        if i == 8:
            ax.set_xlabel('Time bin')
            ax.set_ylabel('CorCE')
        else:
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())

        ax.text(.05, .95, 'Slope = %2i spikes/s^2' % mean_slope,
                transform=ax.transAxes)
        phi_m = np.mean(phis[i, :])
        phi_s = np.std(phis[i, :])
        ax.text(.05, .90, 'Phi = %.2f +/- %.2f' % (phi_m, phi_s),
                transform=ax.transAxes)

    sns.despine(offset=5, trim=True)
    return mean_slopes, phis, corCEs, corCETheory


def plotCorCESimulation(g_counts, n_bins=8):
    fig = plt.figure()
    plot_counter = 1

    # find the best phi
    phis = findBestPhi(
        g_counts['spk_counts'][:, 0:n_bins],
        g_counts['count_means'][:, 0:n_bins],
        np.float64(g_counts['trial_counts'][:, 0:n_bins]),
        min_trials=6)
    g_var = computeVarCorCovCE(
        g_counts['spk_counts'][:, 0:n_bins],
        g_counts['count_means'][:, 0:n_bins],
        g_counts['trial_counts'][:, 0:n_bins],
        phis)
    ax = fig.add_subplot(2, 2, plot_counter)
    cax = ax.imshow(
        g_var['corCE'], vmin=0, vmax=1,
        interpolation='nearest', cmap='jet')
    n_points = g_var['corCE'].shape[0]
    ax.set_xticks(np.arange(n_points))
    ax.set_yticks(np.arange(n_points))
    #ax.set_xticklabels(g_var['times'], rotation=45)
    #ax.set_yticklabels(g_var['times'])
    ax.add_line(plt.Line2D(
        np.arange(n_points)[1:], np.repeat(0, n_points)[1:],
        color='k'))
    ax.add_line(plt.Line2D(
        np.arange(n_points)[0:-1] + 1, np.arange(n_points)[0:-1],
        color='k', linestyle='--'))

    cbar = fig.colorbar(cax, ticks=[0, 0.5, 1])
    # vertically oriented colorbar
    cbar.ax.set_yticklabels(['0.0', '0.5', '1.0'])
    # cbar.ax.set_ylabel('CorCE')

    corCEPred = g_var['corCE']
    n = corCEPred.shape[0]
    corCETheory = corCEpartialSums(n)

    ax = fig.add_subplot(4, 2, plot_counter + 1)
    x = np.arange(2, n + 1)
    ax.scatter(
        x, np.diagonal(corCEPred, 1),
        color='k', s=25, facecolors='none')
    ax.scatter(x, corCEPred[0, 1:], color='k', s=25)
    ax.add_line(
        plt.Line2D(x, np.diagonal(corCETheory, 1),
                   color='k', linestyle='--'))
    ax.add_line(plt.Line2D(x, corCETheory[0, 1:], color='k'))

    misc.adjust_spines(ax, ['left', 'bottom'])
    ax.tick_params(axis='both', direction='out')
    ax.grid('on')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('CorCE')
    #ax.set_xticklabels(g_var['times'])
    ax.set_xlim(2 - .5, n + .5)
    ax.set_ylim(0.3, 1)


def computeFFandPhi(spk_counts, count_means, trial_counts,
                    min_trials=6, prop_cutoff=.5, epsilon=0.01):
    # for a condition to contribute to the grand average, it must have
    # at least min_trials trials remaining at that timepoint;
    # to make the code easier, trials_counts has to be the same shape as
    # spk_counts and count_means;
    # this allows for a more accurate estimation of residuals relative to mean;
    # prop_cutoff refers to the grand matrix, not each condition

    spk_counts = spk_counts.copy()
    count_means = count_means.copy()

    spk_counts[trial_counts < min_trials] = np.nan
    count_means[trial_counts < min_trials] = np.nan

    n_remain = np.sum(~(np.isnan(spk_counts)), axis=0)
    p_terminated = 1 - n_remain / n_remain[0]

    spk_counts = spk_counts[:, p_terminated < prop_cutoff]
    count_means = count_means[:, p_terminated < prop_cutoff]
    residuals = spk_counts - count_means

    var = np.nanstd(residuals, bias=False, axis=0) ** 2
    ff = var / np.nanmean(count_means, axis=0)

    g_out = {}
    if len(ff) == 0:
        g_out['ff'] = np.nan
        g_out['phi'] = np.nan
    else:
        phi = np.nanmin(ff)
        g_out['ff'] = ff
        g_out['phi'] = phi

    return g_out


def nancov(matrix_in):
    C = np.zeros([matrix_in.shape[1], matrix_in.shape[1]])
    N = np.zeros([matrix_in.shape[1], matrix_in.shape[1]])
    for i in xrange(matrix_in.shape[1]):
        for j in np.arange(i, matrix_in.shape[1]):
            v1 = matrix_in[:, i]
            v2 = matrix_in[:, j]
            keep_idx1 = np.where(~np.isnan(v1))[0]
            keep_idx2 = np.where(~np.isnan(v2))[0]
            keep_idx = np.intersect1d(keep_idx1, keep_idx2)
            v1 = v1[keep_idx]
            v2 = v2[keep_idx]
            # cij = np.mean((v1-np.mean(v1))*(v2-np.mean(v2)))
            cij = np.sum(
                (v1 - np.mean(v1)) * (v2 - np.mean(v2))) / (len(v1) - 1)
            C[i, j] = cij
            C[j, i] = cij
            N[[(i, j), (j, i)]] = len(keep_idx)
    return C, N


def fCorVarCEerr(phi, spk_counts, count_means, trial_counts):

    resid = spk_counts - count_means

    # varCE
    varCE = np.nanmean(resid ** 2 - phi * count_means, axis=0)
    n = len(varCE)

    # covCE and corCE
    # masked_counts = np.ma.array(resid, mask=np.isnan(resid))
    # covCE = np.ma.cov(masked_counts, rowvar=False, allow_masked=True)

    covCE = nancov(resid)[0]
    for i in xrange(n):
        covCE[i, i] = varCE[i]

    corCE = np.zeros([n, n])
    corCE[:] = np.nan
    for i in xrange(n):
        for j in xrange(n):
            corCE[i, j] = covCE[i, j] / np.sqrt(covCE[i, i] * covCE[j, j])

    # rmax = np.max(corCE)

    # if np.max(np.abs(corCE))>1:
    #    print 'here abs'
    #    if rmax < lastrmax:
    #        err = .99*lasterr.counter
    #        lasterr(err)
    #    else:
    #        err = 1.01*lasterr.counter
    #        lasterr(err)
    # return err

    rpred = corCEpartialSums(n)

    rCIlo = np.zeros([n, n])
    rCIlo[:] = np.nan
    rCIhi = np.zeros([n, n])
    rCIhi[:] = np.nan

    err = 0
    for i in np.arange(0, n - 1):
        for j in np.arange(i + 1, n):
            n4r = np.min(
                [np.sum(~np.isnan(trial_counts[:, i])),
                 np.sum(~np.isnan(trial_counts[:, j]))])
            zpred = r2z(rpred[i, j])
            zobs = r2z(corCE[i, j])
            s = 1 / np.sqrt(n4r - 3)  # std error
            zCI = zobs + np.array([-1., 1.]) * 1.96 * s
            rCI = (np.exp(2 * zCI) - 1) / (np.exp(2 * zCI) + 1)
            rCIlo[i, j] = rCI[0]
            rCIhi[i, j] = rCI[1]
            err += ((zobs - zpred) / (2 * s)) ** 2  # SS for chi2

            # lasterr(err)
    return err


def findBestPhi(spk_counts, count_means, trial_counts, min_trials=6):
    spk_counts = spk_counts.copy()
    count_means = count_means.copy()
    trial_counts = np.float64(trial_counts.copy())

    # a given condition (coh/direction/cell combo) must have at
    # least min_trials to contribute to analysis at the given timepoint
    spk_counts[trial_counts < min_trials] = np.nan
    count_means[trial_counts < min_trials] = np.nan
    trial_counts[trial_counts < min_trials] = np.nan

    # do an initial grid search for the best phi starting value
    phi_grid = np.arange(.3, 1, .03)
    err_grid = np.zeros_like(phi_grid)
    err_grid[:] = np.nan
    # lasterr(0)
    # lastrmax(0)
    for i in xrange(len(phi_grid)):
        err_grid[i] = fCorVarCEerr(
            phi_grid[i], spk_counts, count_means, trial_counts)
    err_grid_best = np.nanmin(err_grid)
    err_grid_best_idx = np.nanargmin(err_grid)
    phi_grid_best = phi_grid[err_grid_best_idx]

    lasterr(0)
    lastrmax(0)
    phi_fit, err_fit, n_iter, funccall, warnflag = fmin(
        fCorVarCEerr, phi_grid_best,
        args=(spk_counts, count_means, trial_counts),
        full_output=True, disp=False, maxiter=10 ** 6)

    if err_fit > err_grid_best:
        print 'grid outperformed search'
        phi_fit = phi_grid_best
        err_fit = err_grid_mind

    return phi_fit


# retuns the predicted corCE for an unbounded discrete diffusion or random walk
# n is the number of steps
def corCEpartialSums(n):
    R = np.zeros([n, n])
    R[:] = np.nan
    for i in xrange(n):
        for j in xrange(n):
            R[i, j] = np.sqrt(np.min([i + 1, j + 1]) / np.max([i + 1, j + 1]))
    return R


def r2z(r):
    z = .5 * np.log((1 + r) / (1 - r))
    return z


def lasterr(store):
    lasterr.counter = store


def lastrmax(store):
    lastrmax.counter = store


def plotDiffusion():
    m = np.random.randn(100, 200)
    random_walks = np.cumsum(m, axis=1)

    fig = plt.figure()

    ###
    ax = fig.add_subplot(231)
    for i in xrange(random_walks.shape[0]):
        ax.add_line(plt.Line2D(
            np.arange(random_walks.shape[1]), random_walks[i, :],
            color='k', alpha=.35))

    misc.adjust_spines(ax, ['left', 'bottom'])
    ax.tick_params(axis='both', direction='out')
    ax.grid('on')
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('Magnitude (a.u.)')
    ax.set_xlim(0, 100)
    ax.set_ylim(-25, 25)

    ax = fig.add_subplot(234)
    ax.add_line(plt.Line2D(
        np.arange(random_walks.shape[1]), np.var(random_walks, axis=0),
        color='k'))
    misc.adjust_spines(ax, ['left', 'bottom'])
    ax.tick_params(axis='both', direction='out')
    ax.grid('on')
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('Variance')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 125)

    ###
    ax = fig.add_subplot(232)
    for i in xrange(random_walks.shape[0]):
        ax.add_line(plt.Line2D(
            np.arange(random_walks.shape[1]), random_walks[i, :],
            color='k', alpha=.35))

    misc.adjust_spines(ax, ['left', 'bottom'])
    ax.tick_params(axis='both', direction='out')
    ax.grid('on')
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('Magnitude (a.u.)')
    ax.set_xlim(0, 100)
    ax.set_ylim(-25, 25)

    colors = ['red', 'green', 'blue']
    ax.axvline(15, color='black')
    for i in np.arange(3):
        ax.axvline(15 + (i + 1) * 20, color=colors[i])

    ax = fig.add_subplot(235)
    for i in np.arange(3)[::-1]:
        ax.scatter(random_walks[:, 15],
                   random_walks[:, 15 + (i + 1) * 20], color=colors[i],
                   alpha=.2, marker='o')
        ax.text(0.1, 0.9 - i * .05, 'r = %.2f' %
                pearsonr(random_walks[:, 15],
                         random_walks[:, 15 + (i + 1) * 20])[0],
                transform=ax.transAxes, color=colors[i])
    misc.adjust_spines(ax, ['left', 'bottom'])
    ax.tick_params(axis='both', direction='out')
    ax.grid('on')
    ax.set_xlabel('Magnitude (a.u.)')
    ax.set_ylabel('Magnitude (a.u.)')
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)

    ###
    ax = fig.add_subplot(233)
    for i in xrange(random_walks.shape[0]):
        ax.add_line(plt.Line2D(
            np.arange(random_walks.shape[1]), random_walks[i, :],
            color='k', alpha=.35))

    misc.adjust_spines(ax, ['left', 'bottom'])
    ax.tick_params(axis='both', direction='out')
    ax.grid('on')
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('Magnitude (a.u.)')
    ax.set_xlim(0, 100)
    ax.set_ylim(-25, 25)

    ax.axvline(10, color='orange')
    ax.axvline(40, color='orange')
    ax.axvline(30, color='cyan')
    ax.axvline(60, color='cyan')

    ax = fig.add_subplot(236)
    ax.scatter(random_walks[:, 10], random_walks[
               :, 40], color='orange', alpha=.35, marker='o')
    ax.scatter(random_walks[:, 30], random_walks[
               :, 60], color='cyan', alpha=.35, marker='o')
    ax.text(0.1, 0.9, 'r = %.2f' %
            pearsonr(random_walks[:, 10], random_walks[:, 40])[0],
            transform=ax.transAxes, color='orange')
    ax.text(0.1, 0.8, 'r = %.2f' %
            pearsonr(random_walks[:, 30], random_walks[:, 60])[0],
            transform=ax.transAxes, color='cyan')
    misc.adjust_spines(ax, ['left', 'bottom'])
    ax.tick_params(axis='both', direction='out')
    ax.grid('on')
    ax.set_xlabel('Magnitude (a.u.)')
    ax.set_ylabel('Magnitude (a.u.)')
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)

    plt.tight_layout()
