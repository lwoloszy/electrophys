from __future__ import division

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, MultipleLocator
from matplotlib.mlab import movavg
from matplotlib import transforms

import brewer2mpl as b2mpl
import seaborn as sns

sns.set_style('ticks')
sns.set_context('talk')


def getSpkTimesAligned(g, spk_chan, unit_num,
                       t_align, pre_time, post_time):
    n_trials = len(g['spk_times'])
    spk_times = []
    for i in range(n_trials):
        cur_spk_times = np.atleast_1d(np.int64(np.round(
            g['spk_times'][i][spk_chan][unit_num])))
        sel = np.atleast_1d(np.logical_and(
            np.int64(np.round(g['spk_times'][i][spk_chan][unit_num])) >=
            np.int64(np.round(g[t_align][i])) + pre_time,
            np.int64(np.round(g['spk_times'][i][spk_chan][unit_num])) <
            np.int64(np.round(g[t_align][i])) + post_time))
        spk_times.append(
            cur_spk_times[sel] - np.int64(np.round(g[t_align][i])))
    return spk_times


def getSpkTimesBounded(g, spk_chan, unit_num,
                       t_start, t_stop, t_align, pre_offset, post_offset):
    n_trials = len(g['spk_times'])
    spk_times = []
    for i in range(n_trials):
        cur_spk_times = np.atleast_1d(np.int64(np.round(
            g['spk_times'][i][spk_chan][unit_num])))
        sel = np.atleast_1d(np.logical_and(
            np.int64(np.round(g['spk_times'][i][spk_chan][unit_num])) >=
            np.int64(np.round(g[t_start][i])) + pre_offset,
            np.int64(np.round(g['spk_times'][i][spk_chan][unit_num])) <
            np.int64(np.round(g[t_stop][i])) + post_offset))
        spk_times.append(
            cur_spk_times[sel] - np.int64(np.round(g[t_align][i])))
    return spk_times


def getSpksSortedAligned(
        g, spk_chan, unit_num, sort_by, t_align, pre_align, post_align,
        boxcar_width=50, gauss_sigma=25,
        trial_order='rt', ret_spk_bins=True, limit=None):

    depth = lambda L: isinstance(L, list) and max(map(depth, L)) + 1
    if depth(sort_by) != 1:
        raise RuntimeError("Depth of sort_by list has to be 1")

    # Check length of sort_by, function only supports 2 or less sort variables
    if len(sort_by) > 2:
        raise RuntimeError("Number of sort variables can only be 1 or 2")

    # select a subset of trials based on a user-provided boolean selector
    if limit is not None:
        g = selectDataStruct(g, limit)

    g[t_align] = np.int64(np.round(np.float64(g[t_align])))

    n_trials = len(g['spk_times'])

    if len(sort_by) == 1:
        insert_dummy = True
        label1_name = 'dummy variable'
        label2_name = sort_by[0]
        sort_by_list = [g[sort_by[0]]]
    elif len(sort_by) == 2:
        insert_dummy = False
        label1_name = sort_by[0]
        label2_name = sort_by[1]
        sort_by_list = [g[sort_by[0]], g[sort_by[1]]]

    if trial_order == 'rt':
        g['t_trial_order'] = g['t_response'] - g['t_dotson']
    elif trial_order == 'duration':
        g['t_trial_order'] = g['t_response'] - g['t_targetson']
    elif trial_order == 'chronological':
        g['t_trial_order'] = g['t_response']

    trialorder_sorted = sortByLists(g['t_trial_order'], sort_by_list)
    trial_duration = post_align - pre_align

    # psth will be computed aligned on relevant event
    spk_times_aligned = getSpkTimesAligned(
        g, spk_chan, unit_num, t_align, pre_align, post_align
    )
    spk_times_aligned_sorted = sortByLists(
        np.array(spk_times_aligned), sort_by_list
    )

    labels = spk_times_aligned_sorted['labels']
    if insert_dummy:
        labels = np.c_[np.repeat(1, labels.shape[0]), labels]

    u_labels = np.unique(labels[:, 0])
    n_u_labels = len(u_labels)

    all_sorted_trial_counts = list()
    all_sorted_timestamps = list()
    all_sorted_spk_times = list()
    all_sorted_spk_counts = list()
    all_sorted_spk_bins = list()
    all_sorted_times = list()
    all_sorted_frs = list()
    all_sorted_psths = list()
    all_sorted_psth_times = list()

    all_sorted_psths_gauss = list()
    all_sorted_psth_times_gauss = list()

    all_sorted_psths_alpha = list()
    all_sorted_psth_times_alpha = list()

    for i in range(n_u_labels):

        u_sublabels = labels[:, 1][labels[:, 0] == u_labels[i]]

        for j in range(len(u_sublabels)):

            n_trials = np.array(trialorder_sorted['sorted'])[
                np.logical_and(labels[:, 0] == u_labels[i],
                               labels[:, 1] == u_sublabels[j])][0].shape[0]
            if n_trials == 0:
                all_sorted_trial_counts.append(0)
                all_sorted_timestamps.append([])
                all_sorted_spk_times.append([])
                all_sorted_spk_counts.append([])
                all_sorted_spk_bins.append([])
                all_sorted_times.append([])
                all_sorted_psth_times.append([])
                all_sorted_psth_times_gauss.append([])
                all_sorted_psth_times_alpha.append([])
                all_sorted_frs.append([])
                all_sorted_psths.append([])
                all_sorted_psths_gauss.append([])
                all_sorted_psths_alpha.append([])
                continue

            cur_sorted_trialorder = np.array(trialorder_sorted['sorted'])[
                np.logical_and(labels[:, 0] == u_labels[i],
                               labels[:, 1] == u_sublabels[j])]
            cur_sorted_spk_times = np.array(
                spk_times_aligned_sorted['sorted'])[
                np.logical_and(labels[:, 0] == u_labels[i],
                               labels[:, 1] == u_sublabels[j])]

            # Sort by the desired trial order
            cur_sorted_spk_times = cur_sorted_spk_times[0][
                np.argsort(cur_sorted_trialorder[0])]
            cur_sorted_spk_counts = [len(k) for k in cur_sorted_spk_times]
            cur_sorted_frs = [k / (post_align - pre_align) * 1000
                              for k in cur_sorted_spk_counts]

            # We now create a spk_bin array where the rows are trials,
            # each column is a ms, and 0's and 1's indicate the absence
            # and presence of spikes.
            cur_sorted_spk_bin = np.empty([len(cur_sorted_spk_times),
                                           trial_duration])
            cur_sorted_spk_bin[:] = np.nan

            times = np.int64(np.arange(0, trial_duration, 1) + pre_align)
            for k in range(len(cur_sorted_spk_times)):
                cur_sorted_spk_bin[k, 0:trial_duration] = 0
                cur_sorted_spk_bin[
                    k, np.int64(cur_sorted_spk_times[k] - pre_align)] = 1

            # Regular boxcar smoothing
            smoothed_psth = movavg(sp.stats.nanmean(
                cur_sorted_spk_bin, axis=0) / .001, boxcar_width)
            psth_times = movavg(times, boxcar_width) + .5

            # Compute psth smoothed with a gaussian with gauss_sigma
            x = np.arange(0, gauss_sigma * 6 + 1)
            gaussian_exponent = -0.5 * np.power(
                (x - np.mean(x)) / gauss_sigma, 2)
            gaussian_filter = np.exp(gaussian_exponent) / np.sum(
                np.exp(gaussian_exponent))
            smoothed_psth_gauss = np.correlate(
                sp.stats.nanmean(cur_sorted_spk_bin, axis=0) / .001,
                gaussian_filter, mode='valid')
            psth_times_gauss = times[
                int(gauss_sigma * 6 / 2):int(-gauss_sigma * 6 / 2)]

            # Compute psth smoothed with alpha-like function
            t = np.arange(250)
            h = (1 - np.exp(-t)) * np.exp(-t / 25.)
            h = h / np.sum(h)
            smoothed_psth_alpha = np.convolve(
                sp.stats.nanmean(cur_sorted_spk_bin, axis=0) / .001,
                h, mode='full')
            smoothed_psth_alpha = smoothed_psth_alpha[
                0:cur_sorted_spk_bin.shape[1]][25:]
            psth_times_alpha = times[0:cur_sorted_spk_bin.shape[1]][25:]

            all_sorted_trial_counts.append(len(cur_sorted_spk_times))
            all_sorted_spk_times.append(cur_sorted_spk_times)
            all_sorted_spk_counts.append(cur_sorted_spk_counts)
            all_sorted_spk_bins.append(cur_sorted_spk_bin)
            all_sorted_times.append(times)
            all_sorted_frs.append(cur_sorted_frs)
            all_sorted_psths.append(smoothed_psth)
            all_sorted_psth_times.append(psth_times)

            all_sorted_psths_gauss.append(smoothed_psth_gauss)
            all_sorted_psth_times_gauss.append(psth_times_gauss)

            all_sorted_psths_alpha.append(smoothed_psth_alpha)
            all_sorted_psth_times_alpha.append(psth_times_alpha)

    out = {}
    out['label1_name'] = label1_name
    out['label2_name'] = label2_name
    out['label1'] = labels[:, 0]
    out['label2'] = labels[:, 1]
    out['trial_counts'] = all_sorted_trial_counts
    out['spk_times'] = all_sorted_spk_times
    out['spk_counts'] = all_sorted_spk_counts
    out['frs'] = all_sorted_frs
    out['psths'] = all_sorted_psths
    out['psth_times'] = all_sorted_psth_times

    out['gauss_psths'] = all_sorted_psths_gauss
    out['gauss_psth_times'] = all_sorted_psth_times_gauss

    out['alpha_psths'] = all_sorted_psths_alpha
    out['alpha_psth_times'] = all_sorted_psth_times_alpha

    if ret_spk_bins:
        out['spk_bins'] = all_sorted_spk_bins
        out['times'] = all_sorted_times

    return out


def getSpksSortedBounded(
        g, spk_chan, unit_num, sort_by,
        t_start, t_stop, t_align, pre_offset, post_offset,
        cutoff=.5, boxcar_width=50, gauss_sigma=25,
        trial_order='rt', ret_spk_bins=True, limit=None):

    depth = lambda L: isinstance(L, list) and max(map(depth, L)) + 1
    if depth(sort_by) != 1:
        raise RuntimeError("Depth of sort_by list has to be 1")

    # Check length of sort_by, function only supports 2 or less sort variables
    if len(sort_by) > 2:
        raise RuntimeError("Number of sort variables can only be 1 or 2")

    # Check that t_start and t_stop are feasible
    if (g[t_stop] - g[t_start])[0] < 0:
        raise RuntimeError("t_stop needs to be a timestamp after t_start")

    # Select a subset of trials based on a user-provided boolean selector
    if limit is not None:
        g = selectDataStruct(g, limit)

    g[t_start] = np.int64(np.round(np.float64(g[t_start])))
    g[t_stop] = np.int64(np.round(np.float64(g[t_stop])))

    # Get rid of trials that do not "fit" into the time window requested
    durations = g[t_stop] - g[t_start]
    keep_boolean = np.repeat(True, len(durations))
    for i in range(len(durations)):
        if (g[t_stop][i] + post_offset) - (g[t_start][i] + pre_offset) <= 0:
            keep_boolean[i] = False
    g = selectDataStruct(g, keep_boolean)

    n_trials = len(g['spk_times'])
    trial_ids = np.arange(n_trials)

    if len(sort_by) == 1:
        insert_dummy = True
        label1_name = 'dummy variable'
        label2_name = sort_by[0]
        sort_by_list = [g[sort_by[0]]]
    elif len(sort_by) == 2:
        insert_dummy = False
        label1_name = sort_by[0]
        label2_name = sort_by[1]
        sort_by_list = [g[sort_by[0]], g[sort_by[1]]]

    if trial_order == 'rt':
        g['t_trial_order'] = g['t_response'] - g['t_dotson']
    elif trial_order == 'duration':
        g['t_trial_order'] = g['t_response'] - g['t_targetson']
    elif trial_order == 'chronological':
        g['t_trial_order'] = g['t_response']
    trialorder_sorted = sortByLists(g['t_trial_order'], sort_by_list)
    trialids_sorted = sortByLists(trial_ids, sort_by_list)

    trial_durations = [(g[t_stop][i] + post_offset) -
                       (g[t_start][i] + pre_offset)
                       for i in range(n_trials)]
    trial_durations_sorted = sortByLists(
        np.array(trial_durations), sort_by_list)

    # To ensure that each trial has at least one NaN, we will add 1 msec
    # to max_trial_duration, as that simplifies the calculations below
    max_trial_duration = np.int64(np.max(trial_durations)) + 1

    # Timestamps
    if t_align == t_start:
        timestamps = [g[t_stop][i] - g[t_align][i] + post_offset
                      for i in range(n_trials)]
        timestamps_sorted = sortByLists(
            np.array(timestamps), sort_by_list)
    elif t_align == t_stop:
        timestamps = [g[t_start][i] - g[t_align][i] + pre_offset
                      for i in range(n_trials)]
        timestamps_sorted = sortByLists(
            np.array(timestamps), sort_by_list)

    # PSTHs will be computed with attrition
    spk_times_bounded = getSpkTimesBounded(
        g, spk_chan, unit_num,
        t_start, t_stop, t_align, pre_offset, post_offset)
    spk_times_bounded_sorted = sortByLists(
        np.array(spk_times_bounded), sort_by_list)

    labels = spk_times_bounded_sorted['labels']
    if insert_dummy:
        labels = np.column_stack([np.repeat(1, labels.shape[0]), labels])

    u_labels = np.unique(labels[:, 0])
    n_u_labels = len(u_labels)

    all_sorted_timestamps = list()
    all_sorted_trialids = list()
    all_sorted_trial_counts = list()
    all_sorted_spk_times = list()
    all_sorted_spk_counts = list()
    all_sorted_spk_bins = list()
    all_sorted_times = list()
    all_sorted_chopped_times = list()
    all_sorted_frs = list()
    all_sorted_psths = list()

    all_sorted_psths_gauss = list()
    all_sorted_chopped_times_gauss = list()

    all_sorted_psths_alpha = list()
    all_sorted_chopped_times_alpha = list()

    all_sorted_chopped_trial_counts = list()

    for i in range(n_u_labels):

        u_sublabels = labels[:, 1][labels[:, 0] == u_labels[i]]

        for j in range(len(u_sublabels)):

            n_trials = np.array(trial_durations_sorted['sorted'])[
                np.logical_and(labels[:, 0] == u_labels[i],
                               labels[:, 1] == u_sublabels[j])][0].shape[0]

            if n_trials == 0:
                all_sorted_trial_counts.append(0)
                all_sorted_trialids.append([])
                all_sorted_timestamps.append([])
                all_sorted_spk_times.append([])
                all_sorted_spk_counts.append([])
                all_sorted_spk_bins.append([])
                all_sorted_times.append([])
                all_sorted_chopped_times.append([])
                all_sorted_chopped_times_gauss.append([])
                all_sorted_chopped_times_alpha.append([])
                all_sorted_chopped_trial_counts.append([])
                all_sorted_frs.append([])
                all_sorted_psths.append([])
                all_sorted_psths_gauss.append([])
                all_sorted_psths_alpha.append([])
                continue

            sel = np.logical_and(labels[:, 0] == u_labels[i],
                                 labels[:, 1] == u_sublabels[j])

            cur_sorted_trialorder = np.array(trialorder_sorted['sorted'])[sel]

            cur_sorted_trialids = np.array(trialids_sorted['sorted'])[sel]
            cur_sorted_trial_durations = \
                np.array(trial_durations_sorted['sorted'])[sel]
            cur_sorted_spk_times = \
                np.array(spk_times_bounded_sorted['sorted'])[sel]
            cur_sorted_timestamps = np.array(timestamps_sorted['sorted'])[sel]

            cur_sorted_trialids = cur_sorted_trialids[0][
                np.argsort(cur_sorted_trialorder[0])]
            cur_sorted_trial_durations = cur_sorted_trial_durations[0][
                np.argsort(cur_sorted_trialorder[0])]
            cur_sorted_spk_times = cur_sorted_spk_times[0][
                np.argsort(cur_sorted_trialorder[0])]
            cur_sorted_spk_counts = [len(k) for k in cur_sorted_spk_times]
            cur_sorted_frs = [
                cur_sorted_spk_counts[k] / cur_sorted_trial_durations[k] * 1000
                for k in range(len(cur_sorted_spk_counts))]
            cur_sorted_timestamps = cur_sorted_timestamps[0][
                np.argsort(cur_sorted_trialorder[0])]

            # We now create a spk_bin array where the rows are trials, each
            # column is 1 ms, and 0's and 1's indicate the absence and presence
            # of spikes. Nan's indicate that the trial does not extend that far
            # relative to the user-specified align events and offset times.
            # Working in ms!!!
            cur_sorted_spk_bin = np.empty([len(cur_sorted_spk_times),
                                           max_trial_duration])
            cur_sorted_spk_bin[:] = np.nan

            if t_align == t_start:

                times = np.int64(
                    np.arange(0, max_trial_duration, 1)) + pre_offset
                for k in range(len(cur_sorted_spk_times)):
                    cur_sorted_spk_bin[
                        k, 0:np.int64(cur_sorted_trial_durations[k])] = 0
                    cur_sorted_spk_bin[
                        k, np.int64(cur_sorted_spk_times[k] - pre_offset)] = 1
                try:
                    median_cutoff = np.where(
                        np.sum(np.isnan(cur_sorted_spk_bin), axis=0) /
                        n_trials > cutoff)[0][0]
                except IndexError:
                    return 'Goddamn handling of cutoff throwing some weird bug'

                chopped_spk_bin = cur_sorted_spk_bin[:, 0:median_cutoff]
                chopped_times = times[0:median_cutoff]
                chopped_trial_counts = np.sum(
                    ~np.isnan(chopped_spk_bin), axis=0)

            elif t_align == t_stop:
                times = np.int64(np.arange(
                    post_offset - max_trial_duration, post_offset))
                for k in range(len(cur_sorted_spk_times)):
                    cur_sorted_spk_bin[
                        k, -np.int64(cur_sorted_trial_durations[k]):] = 0
                    cur_sorted_spk_bin[
                        k, np.int64(cur_sorted_spk_times[k] - post_offset)] = 1
                try:
                    median_cutoff = np.where(
                        np.sum(np.isnan(cur_sorted_spk_bin), axis=0) /
                        n_trials > cutoff)[0][-1] + 1
                except IndexError:
                    print 'Goddamn handling of cutoff throwing some weird bug'

                chopped_spk_bin = cur_sorted_spk_bin[:, median_cutoff:]
                chopped_times = times[median_cutoff:]
                chopped_trial_counts = np.sum(
                    ~np.isnan(chopped_spk_bin), axis=0)

            # Regular boxcar smoothing
            smoothed_psth = movavg(sp.stats.nanmean(
                chopped_spk_bin, axis=0) / .001, boxcar_width)

            # Gaussian smoothing with gauss_sigma
            x = np.arange(0, gauss_sigma * 6 + 1)
            gaussian_exponent = -0.5 * np.power(
                (x - np.mean(x)) / gauss_sigma, 2)
            gaussian_filter = np.exp(
                gaussian_exponent) / np.sum(np.exp(gaussian_exponent))
            smoothed_psth_gauss = np.convolve(
                sp.stats.nanmean(chopped_spk_bin, axis=0) / .001,
                gaussian_filter, mode='valid')

            # Alpha-like functions smoothing
            t = np.arange(250)
            h = (1 - np.exp(-t)) * np.exp(-t / 25.)
            h = h / np.sum(h)
            smoothed_psth_alpha = np.convolve(
                sp.stats.nanmean(chopped_spk_bin, axis=0) / .001,
                h, mode='full')
            smoothed_psth_alpha = smoothed_psth_alpha[
                0:chopped_spk_bin.shape[1]][25:]
            psth_times_alpha = chopped_times[0:chopped_spk_bin.shape[1]][25:]

            all_sorted_timestamps.append(cur_sorted_timestamps)
            all_sorted_trial_counts.append(len(cur_sorted_spk_times))
            all_sorted_trialids.append(cur_sorted_trialids)
            all_sorted_spk_times.append(cur_sorted_spk_times)
            all_sorted_spk_counts.append(cur_sorted_spk_counts)
            all_sorted_spk_bins.append(cur_sorted_spk_bin)
            all_sorted_times.append(times)
            all_sorted_chopped_times.append(
                movavg(chopped_times, boxcar_width) + .5)
            all_sorted_chopped_times_gauss.append(chopped_times[
                int(gauss_sigma * 6 / 2):int(-gauss_sigma * 6 / 2)])
            all_sorted_chopped_times_alpha.append(psth_times_alpha)
            all_sorted_chopped_trial_counts.append(chopped_trial_counts)
            all_sorted_frs.append(cur_sorted_frs)
            all_sorted_psths.append(smoothed_psth)
            all_sorted_psths_gauss.append(smoothed_psth_gauss)
            all_sorted_psths_alpha.append(smoothed_psth_alpha)

    out = {}

    out['label1_name'] = label1_name
    out['label2_name'] = label2_name
    out['label1'] = labels[:, 0]
    out['label2'] = labels[:, 1]
    out['timestamps'] = all_sorted_timestamps
    out['trial_counts'] = all_sorted_trial_counts
    out['trial_ids'] = all_sorted_trialids
    out['spk_times'] = all_sorted_spk_times
    out['spk_counts'] = all_sorted_spk_counts
    out['chopped_times'] = all_sorted_chopped_times
    out['chopped_trial_counts'] = all_sorted_chopped_trial_counts
    out['frs'] = all_sorted_frs
    out['psths'] = all_sorted_psths

    out['gauss_psths'] = all_sorted_psths_gauss
    out['gauss_chopped_times'] = all_sorted_chopped_times_gauss

    out['alpha_psths'] = all_sorted_psths_alpha
    out['alpha_chopped_times'] = all_sorted_chopped_times_alpha

    if trial_order == 'rt':
        out['rts_sorted'] = trialorder_sorted

    if ret_spk_bins:
        out['spk_bins'] = all_sorted_spk_bins
        out['times'] = all_sorted_times

    return out


def plotSpksAligned(
        g, spk_chan, unit_num, sort_by,
        t_align, pre_align, post_align,
        smooth_type='boxcar', boxcar_width=50, gauss_sigma=25,
        trial_order='rt',
        colormap='Set1', colormap_type='Qualitative', reverse_colors=False,
        limit=None, close_all=False):

    g_anal = getSpksSortedAligned(
        g, spk_chan, unit_num,
        sort_by, t_align, pre_align, post_align,
        boxcar_width=boxcar_width, gauss_sigma=gauss_sigma,
        trial_order=trial_order, limit=limit)

    n_trials = len(g['spk_times'])

    if close_all:
        plt.close('all')

    fig = plt.figure()
    fig.subplots_adjust(wspace=0.04, hspace=0.00)

    u_labels = np.unique(g_anal['label1'])
    n_u_labels = len(u_labels)

    # Maximum number of trials in one panel,
    # where each panel is a unique label1.
    # This will be used to scale tick marks
    max_panel_count = np.max(
        [np.sum(np.array(g_anal['trial_counts'])[g_anal['label1'] == i])
         for i in np.unique(g_anal['label1'])])

    for i in range(n_u_labels):
        axRasters = fig.add_subplot(2, n_u_labels, i + 1)
        axPSTH = fig.add_subplot(2, n_u_labels, i + 1 + n_u_labels)

        axRasters.set_ylim(0, 1)
        pixels_per_yunit = (axRasters.transData.transform_point((1, 1)) -
                            axRasters.transData.transform((0, 0)))[1]

        panel_trl_ctr = 0
        panel_color_ctr = 0

        u_sublabels = g_anal['label2'][g_anal['label1'] == u_labels[i]]
        n_colors = len(u_sublabels)
        colors = b2mpl.get_map(
            colormap, colormap_type, n_colors).mpl_colors
        if reverse_colors:
            colors = colors[::-1]

        for j in range(len(u_sublabels)):

            n_trials = np.array(g_anal['trial_counts'])[
                np.logical_and(g_anal['label1'] == u_labels[i],
                               g_anal['label2'] == u_sublabels[j])][0]
            if not(n_trials):  # zero trials in this condition
                axPSTH.add_line(
                    plt.Line2D([np.nan], [np.nan],
                               color=colors[panel_color_ctr]))
                panel_color_ctr += 1
                continue

            sel = np.logical_and(g_anal['label1'] == u_labels[i],
                                 g_anal['label2'] == u_sublabels[j])

            times = np.array(g_anal['times'])[sel][0]
            spk_bin = np.array(g_anal['spk_bins'])[sel][0]

            if smooth_type == 'boxcar':
                psth_times = np.array(g_anal['psth_times'])[sel][0]
                psth = np.array(g_anal['psths'])[sel][0]
            elif smooth_type == 'gaussian':
                psth_times = np.array(g_anal['gauss_psth_times'])[sel][0]
                psth = np.array(g_anal['gauss_psths'])[sel][0]
            elif smooth_type == 'alpha':
                psth_times = np.array(g_anal['alpha_psth_times'])[sel][0]
                psth = np.array(g_anal['alpha_psths'])[sel][0]
            else:
                raise RuntimeError("Incorrect smooth type")

            # Now create the raster plots
            # We need at least one bin or else we'll get an error
            # when setting axis limits, hence the max operator
            n_total_spikes = np.max([np.sum(spk_bin == 1), 1])
            xdataRasters = np.zeros(n_total_spikes)
            ydataRasters = np.zeros(n_total_spikes)

            offset = 0
            cur_n_trials = spk_bin.shape[0]

            for k in range(cur_n_trials):
                n_spikes = np.sum(spk_bin[k, :] == 1)
                spk_times = times[spk_bin[k, :] == 1]
                xdataRasters[offset:(offset + n_spikes)] = spk_times
                ydataRasters[offset:(offset + n_spikes)] = np.repeat(
                    axRasters.get_ylim()[1] - .001 - panel_trl_ctr /
                    (max_panel_count + 1), n_spikes)
                offset += n_spikes
                panel_trl_ctr += 1
            axRasters.add_line(plt.Line2D(
                xdataRasters, ydataRasters, linestyle='None', marker='|',
                markersize=np.ceil(pixels_per_yunit / max_panel_count),
                markeredgewidth=1.25, color=colors[panel_color_ctr]))
            axPSTH.add_line(plt.Line2D(
                psth_times, psth,
                color=colors[panel_color_ctr]))
            panel_color_ctr += 1

        axRasters.axvline(0, color='k', linestyle='--')
        axPSTH.axvline(0, color='k', linestyle='--')
        axPSTH.set_xlabel('Time (ms)')

        if i == 0:
            axPSTH.set_ylabel('Firing rate (spikes per s)')
        else:
            axPSTH.yaxis.set_major_formatter(NullFormatter())
        sns.despine(ax=axRasters, bottom=True, left=True)
        axRasters.set_xticks([])
        axRasters.set_yticks([])

        if isinstance(u_labels[i], str):
            axRasters.set_title(
                '%s = %s\n\n' % (g_anal['label1_name'], u_labels[i]))
        else:
            axRasters.set_title(
                '%s = %.3f\n\n' % (g_anal['label1_name'], u_labels[i]))
        transform = transforms.blended_transform_factory(
            axRasters.transData, axRasters.transAxes)
        axRasters.text(
            0, 1.15, '%s' % t_align, transform=transform, rotation=30,
            color='k')

    lines = axPSTH.get_lines()
    leg = fig.legend(
        lines, u_sublabels, labelspacing=.05, handlelength=1,
        handletextpad=.3, loc=7)
    texts = leg.get_texts()
    for t, c in zip(texts, colors):
        t.set_color(c)
    plt.show()

    axes = fig.get_axes()
    psth_axes = axes[1::2]

    # set appropriate x and y limits for the plots
    axes = fig.get_axes()
    psth_axes = axes[1::2]
    ymax = -np.inf
    for ax in psth_axes:
        cur_ymax = np.nanmax([np.nanmax(i.get_ydata())
                              for i in ax.get_lines()])
        if cur_ymax > ymax:
            ymax = cur_ymax

    xmin = np.inf
    xmax = -np.inf
    for ax in psth_axes:
        cur_xmin = np.nanmin(
            [np.nanmin(i.get_xdata()) for i in ax.get_lines()])
        cur_xmax = np.nanmax(
            [np.nanmax(i.get_xdata()) for i in ax.get_lines()])
        if cur_xmin < xmin:
            xmin = cur_xmin
        if cur_xmax > xmax:
            xmax = cur_xmax

    for ax in axes:
        ax.set_xlim(xmin, xmax)

    for ax in psth_axes:
        r = xmax - xmin
        majorLocator = MultipleLocator(np.round(r / 4. / 200.0) * 200.0)
        ax.xaxis.set_major_locator(majorLocator)
        ax.set_ylim(0, ymax * 1.05)
        sns.despine(ax=ax, offset=5)

    plt.tight_layout(rect=[0, 0, .925, 1.0])


def plotSpksBounded(
        g, spk_chan, unit_num, sort_by,
        t_start, t_stop, t_align, pre_offset, post_offset,
        cutoff=0.5, boxcar_width=50, gauss_sigma=25,
        trial_order='rt', smooth_type='boxcar',
        colormap_type='Qualitative', colormap='Set1', reverse_colors=False,
        close_all=True, limit=None):

    g_anal = getSpksSortedBounded(
        g, spk_chan, unit_num, sort_by,
        t_start, t_stop, t_align, pre_offset, post_offset,
        cutoff=cutoff, boxcar_width=boxcar_width, gauss_sigma=gauss_sigma,
        trial_order=trial_order, limit=limit)

    n_trials = len(g['spk_times'])

    if close_all:
        plt.close('all')
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.04, hspace=0.00)

    u_labels = np.unique(g_anal['label1'])
    n_u_labels = len(u_labels)

    # Maximum number of trials in one panel,
    # if one_panel is False, each panel is a unique label1
    # if one_panel is True, there will only be one panel
    max_panel_count = np.max(
        [np.sum(np.array(g_anal['trial_counts'])[g_anal['label1'] == i])
         for i in np.unique(g_anal['label1'])])

    for i in range(n_u_labels):
        axRasters = fig.add_subplot(2, n_u_labels, i + 1)
        axPSTH = fig.add_subplot(2, n_u_labels, i + 1 + n_u_labels)

        axRasters.set_ylim(0, 1)
        pixels_per_yunit = (axRasters.transData.transform_point((1, 1)) -
                            axRasters.transData.transform((0, 0)))[1]

        panel_trl_ctr = 0
        panel_color_ctr = 0

        u_sublabels = g_anal['label2'][g_anal['label1'] == u_labels[i]]
        n_colors = np.min([np.max([len(u_sublabels), 3]), 11])
        colors = b2mpl.get_map(
            colormap, colormap_type, n_colors).mpl_colors

        for j in range(len(u_sublabels)):

            n_trials = np.array(g_anal['trial_counts'])[
                np.logical_and(g_anal['label1'] == u_labels[i],
                               g_anal['label2'] == u_sublabels[j])][0]
            if not(n_trials):
                # Add a dummy line
                axPSTH.add_line(plt.Line2D(
                    [np.nan], [np.nan], color=colors[panel_color_ctr]))
                panel_color_ctr += 1
                continue

            sel = np.logical_and(g_anal['label1'] == u_labels[i],
                                 g_anal['label2'] == u_sublabels[j])
            timestamps = np.array(g_anal['timestamps'])[sel][0]
            times = np.array(g_anal['times'])[sel][0]
            spk_bin = np.array(g_anal['spk_bins'])[sel][0]

            if smooth_type == 'boxcar':
                chopped_times = np.array(g_anal['chopped_times'])[sel][0]
                psth = np.array(g_anal['psths'])[sel][0]
            elif smooth_type == 'gaussian':
                chopped_times = np.array(g_anal['gauss_chopped_times'])[sel][0]
                psth = np.array(g_anal['gauss_psths'])[sel][0]
            elif smooth_type == 'alpha':
                chopped_times = np.array(g_anal['alpha_chopped_times'])[sel][0]
                psth = np.array(g_anal['alpha_psths'])[sel][0]
            else:
                raise RuntimeError("Incorrect smooth type")

            # Wow create the raster plots
            # We need at least one bin or else we'll get an error
            # when setting axis limits, hence the max operator
            n_total_spikes = np.max([np.sum(spk_bin == 1), 1])
            xdataRasters = np.zeros(n_total_spikes)
            ydataRasters = np.zeros(n_total_spikes)

            cur_n_trials = spk_bin.shape[0]

            xdataTimestamps = np.zeros(cur_n_trials)
            ydataTimestamps = np.zeros(cur_n_trials)

            offset = 0
            for k in range(cur_n_trials):
                n_spikes = np.sum(spk_bin[k, :] == 1)
                spk_times = times[spk_bin[k, :] == 1]
                xdataRasters[offset:(offset + n_spikes)] = spk_times
                ydataRasters[offset:(offset + n_spikes)] = np.repeat(
                    1 / (max_panel_count + 1) + panel_trl_ctr /
                    (max_panel_count + 1), n_spikes)
                xdataTimestamps[k] = timestamps[k]
                ydataTimestamps[k] = 1 / (max_panel_count + 1) + \
                    panel_trl_ctr / (max_panel_count + 1)

                offset += n_spikes
                panel_trl_ctr += 1

            axRasters.add_line(plt.Line2D(
                xdataRasters, ydataRasters, linestyle='None', marker='|',
                markersize=np.ceil(pixels_per_yunit / max_panel_count),
                markeredgewidth=1.25, color=colors[panel_color_ctr]))

            if t_align == t_start:
                axRasters.add_line(plt.Line2D(
                    xdataTimestamps, ydataTimestamps, linestyle='None',
                    color='k', marker='<'))
            elif t_align == t_stop:
                axRasters.add_line(plt.Line2D(
                    xdataTimestamps, ydataTimestamps, linestyle='None',
                    color='k', marker='>'))

            axPSTH.add_line(plt.Line2D(
                chopped_times, psth,
                color=colors[panel_color_ctr]))

            panel_color_ctr += 1

        axRasters.axvline(0, color='k', linestyle='--')
        axPSTH.axvline(0, color='k', linestyle='--')
        axPSTH.set_xlabel('Time (ms)')

        if i == 0:
            axPSTH.set_ylabel('Firing rate (spikes per s)')
        else:
            axPSTH.yaxis.set_major_formatter(NullFormatter())
        sns.despine(ax=axRasters, bottom=True, left=True)
        axRasters.set_xticks([])
        axRasters.set_yticks([])

        if isinstance(u_labels[i], str):
            axRasters.set_title(
                '%s = %s\n\n' % (g_anal['label1_name'], u_labels[i]))
        else:
            axRasters.set_title(
                '%s = %.3f\n\n' % (g_anal['label1_name'], u_labels[i]))
        transform = transforms.blended_transform_factory(
            axRasters.transData, axRasters.transAxes)
        axRasters.text(
            0, 1.15, '%s' % t_align, transform=transform, rotation=30,
            color='k')

    lines = axPSTH.get_lines()
    leg = fig.legend(
        lines, u_sublabels, labelspacing=.05, handlelength=1,
        handletextpad=.3, loc=7)
    texts = leg.get_texts()
    for t, c in zip(texts, colors):
        t.set_color(c)
    plt.show()

    # set appropriate x and y limits for the plots
    axes = fig.get_axes()
    psth_axes = axes[1::2]
    ymax = -np.inf
    for ax in psth_axes:
        cur_ymax = np.nanmax([np.nanmax(i.get_ydata())
                              for i in ax.get_lines()])
        if cur_ymax > ymax:
            ymax = cur_ymax

    xmin = np.inf
    xmax = -np.inf
    for ax in psth_axes:
        cur_xmin = np.nanmin(
            [np.nanmin(i.get_xdata()) for i in ax.get_lines()])
        cur_xmax = np.nanmax(
            [np.nanmax(i.get_xdata()) for i in ax.get_lines()])
        if cur_xmin < xmin:
            xmin = cur_xmin
        if cur_xmax > xmax:
            xmax = cur_xmax

    for ax in axes:
        ax.set_xlim(xmin, xmax)

    for ax in psth_axes:
        r = xmax - xmin
        majorLocator = MultipleLocator(np.round(r / 4. / 200.0) * 200.0)
        ax.xaxis.set_major_locator(majorLocator)
        ax.set_ylim(0, ymax * 1.05)
        sns.despine(ax=ax, offset=5)

    plt.tight_layout(rect=[0, 0, .925, 1.0])


# Utility function to compute spike rates smoothed with boxcar
# of width bin_size and stepped every step_size.
#
# Spk_bins is a n*m array with n the number of trials and m
# the maximum length of any trial in the array.
#
# 0/1 indicates absence/presence of a spike and
# np.nans show that the trial has terminated
#
# Times is array of length m

def computeBinnedRates(
        spk_bins, times, step_size=5, bin_size=50, direction='forward'):

    if direction == 'forward':
        time_windows = np.c_[
            np.arange(times[0], times[-1] - bin_size + 1, step_size),
            np.arange(times[0], times[-1] - bin_size + 1,
                      step_size) + bin_size]
    elif direction == 'backward':
        time_windows = np.c_[
            np.arange(times[-1], times[0] + bin_size - 1,
                      -step_size) - bin_size,
            np.arange(times[-1], times[0] + bin_size - 1, -step_size)]

    n_trials = spk_bins.shape[0]
    n_windows = time_windows.shape[0]

    spk_counts = np.zeros([n_trials, n_windows])
    spk_counts[:] = np.nan

    for win_num in range(n_windows):
        if direction == 'forward':
            spk_counts[:, win_num] = np.sum(
                spk_bins[:, np.logical_and(
                    times >= time_windows[win_num][0],
                    times < time_windows[win_num][1])], axis=1)
            bin_centers = np.mean(time_windows, axis=1)
        elif direction == 'backward':
            spk_counts[:, -(win_num + 1)] = np.sum(
                spk_bins[:, np.logical_and(
                    times >= time_windows[win_num][0],
                    times < time_windows[win_num][1])], axis=1)
            bin_centers = np.mean(time_windows, axis=1)[::-1]
    spk_rates = spk_counts / bin_size * 1000
    return spk_rates, bin_centers


##############################################################################

def selectDataStruct(validDataStruct, boolSelect):
    selectDataStruct = {}
    for key in validDataStruct.keys():
        try:
            selectDataStruct[key] = np.array(validDataStruct[key])[boolSelect]
        except IndexError:
            print 'something"s the wrong shape'
    return selectDataStruct


def sortByLists(x, ys):
    uniques = [np.sort(np.unique(y)) for y in ys]
    combinations = cartesian(uniques)
    sortedX = list()
    counts = list()
    for i in xrange(combinations.shape[0]):
        selector = list()
        for j in xrange(combinations.shape[1]):
            selector.append(np.array(ys[j]) == combinations[i][j])
        # means all entries matched
        selector = np.sum(selector, axis=0) == combinations.shape[1]
        sortedX.append(np.array(x)[selector])
        counts.append(np.sum(selector))

    out = dict()
    out['sorted'] = sortedX
    out['labels'] = combinations
    out['counts'] = counts
    return out

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """
    arrays = [np.asarray(x) for x in arrays]

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=object)

    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in xrange(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out
