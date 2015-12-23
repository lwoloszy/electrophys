from __future__ import division

# graphing stuff
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from matplotlib.mlab import movavg
import Tkinter as Tk

# communications with plexdll and analysis stuff
import ctypes
from collections import defaultdict
import numpy as np

matplotlib.use('TkAgg')

MAX_EVENTS_PER_READ = 10000
NCOHERENCES = 11
NPLOTS = 9
NUNITS = 20

PLOTPRETIME = 500
PLOTPOSTTIME = 500

COLORS = ['green', 'magenta', 'black'] # last one is a dummy

# relevant ecodes spit out by Rex
LookupTable = {
    'TASKID': 8017,

    'TRIALBEG': 1005,
    'TRIALEND': 4925,
    'CORRECT': 4905,
    'WRONG': 4906,
    'NOCHOICE': 4907,
    'FIXBREAK': 4955,
    'FIXHANDBREAK': 4956,
    'DIRECTION': 8010,
    'COHERENCE': 8011,
    'TARG1_X': 8008,
    'TARG2_X': 8012,

    'SACMADE': 1007,
    'REACHMADE': 1008,

    'FIX_ON': 1010,
    'TARGETS_ON': 3001,
    'DOTS_ON': 4901,
    'DOTS_OFF': 1101,
    'FIX_OFF': 1025,
    'SACCADE_DETECTED': 1007,
    'REACH_DETECTED': 1008,
    'TARGET_ACQUIRED': 4919,
}

ResponseTable = {
    1: 'CORRECT',
    0: 'WRONG',
    -1: 'NOCHOICE',
    -2: 'FIXBREAK',
    -3: 'FIXHANDBREAK',
    -4: 'DID NOT INIT',
}


# define the Plexon event structure (ctypes)
class PL_Event(ctypes.Structure):
    _fields_ = [("Type", ctypes.c_char),
                ("NumberofBlocksInRecord", ctypes.c_char),
                ("BlockNumberInRecord", ctypes.c_char),
                ("UpperTS", ctypes.c_ubyte),
                ("TimeStamp", ctypes.c_ulong),
                ("Channel", ctypes.c_short),
                ("Unit", ctypes.c_short),
                ("DataType", ctypes.c_char),
                ("NumberOfBlocksPerWaveform", ctypes.c_char),
                ("BlockNumberForWaveform", ctypes.c_char),
                ("NumberOfDataWords", ctypes.c_char)]


class DataGatherer(object):
    """
    For storing relevant events collected from Plexon
    """
    def __init__(self):
        # load the Plexon dll
        try:
            self.plexdll = ctypes.windll.LoadLibrary("PlexClient.dll")
        except WindowsError:
            raise RuntimeError(
                "Could not load PlexClient.dll; make sure it's on your path")

        # establish connection with MAP Server
        self.plexdll.PL_InitClientEx3(0, None, None)

        # check if sort client is running, if not quit
        if not(self.plexdll.PL_IsSortClientRunning()):
            raise RuntimeError(
                "Sort Client appears not to be running; no data to gather")

        self.stimInfo = defaultdict(list)
        self.tempSpkHolder = defaultdict(list)

        self.inTrial = False
        self.gotResponse = False
        self.atleast_onevalid = False

        self.got_t_fixon = False
        self.got_t_fixoff = False
        self.got_t_targetson = False
        self.got_t_dotson = False
        self.got_t_dotsoff = False
        self.got_t_response = False

        self.got_dotdirection = False

        # when left hanging from a single query
        self.waitingForDir = False
        self.waitingForCoh = False
        self.waitingForTarg1X = False
        self.waitingForTarg2X = False
        self.waitingForTaskID = False

    def getMAPEvents(self):
        self.trialParsed = False

        # max number of events to read
        nEvents = ctypes.c_int(MAX_EVENTS_PER_READ)

        # allocate memory (array of structures)
        PLEvents = (PL_Event * MAX_EVENTS_PER_READ)()

        # get data from MAP server
        self.plexdll.PL_GetTimeStampStructures(ctypes.byref(nEvents), PLEvents)

        # get the strobed events and spike times
        strobedEvents = list()
        strobedEventTimes = list()
        for i in range(nEvents.value):
            if ((PLEvents[i].Channel == 257) & (PLEvents[i].Unit >= 0)):
                strobedEvents.append(PLEvents[i].Unit)
                strobedEventTimes.append(PLEvents[i].TimeStamp / 40000. * 1000)
            if (int(PLEvents[i].Type.encode('hex')) == 1):
                self.tempSpkHolder['TimeStamps'].append(
                    PLEvents[i].TimeStamp / 40000. * 1000)
                self.tempSpkHolder['Channels'].append(PLEvents[i].Channel)
                self.tempSpkHolder['Units'].append(PLEvents[i].Unit)

        # The following takes care of "hanging" values, that is, values
        # we should have gotten on the last query of the MAP server
        # This is required for event codes that are sent as "pairs"
        if self.waitingForDir is True:
            self.stimInfo['direction'].append(strobedEvents[0] - 7500)
            self.waitingForDir = False
            self.got_dotdirection = True
        if self.waitingForCoh is True:
            self.stimInfo['coherence'].append((strobedEvents[0] - 7000) / 10.)
            self.waitingForCoh = False
        if self.waitingForTarg1X is True:
            self.stimInfo['targ1x'].append((strobedEvents[0] - 7500) / 10.)
            self.waitingForTarg1X = False
        if self.waitingForTarg2X is True:
            self.stimInfo['targ2x'].append((strobedEvents[0] - 7500) / 10.)
            self.waitingForTarg2X = False
        if self.waitingForTaskID is True:
            self.stimInfo['taskid'].append(strobedEvents[0] - 7000)
            self.waitingForTaskID = False

        # Parse events into trial format
        nStrobedEvents = len(strobedEvents)
        for i in range(nStrobedEvents):
            if not(self.inTrial):
                if strobedEvents[i] == LookupTable['TRIALBEG']:
                    self.inTrial = True
                    self.stimInfo['t_begin'].append(strobedEventTimes[i])
            if self.inTrial:
                if (strobedEvents[i] == LookupTable['DIRECTION']):
                    if i == (nStrobedEvents - 1):
                        self.waitingForDir = True
                    else:
                        self.stimInfo['direction'].append(
                            strobedEvents[i + 1] - 7500)
                        self.got_dotdirection = True
                if (strobedEvents[i] == LookupTable['COHERENCE']):
                    if i == (nStrobedEvents - 1):
                        self.waitingForCoh = True
                    else:
                        self.stimInfo['coherence'].append(
                            (strobedEvents[i + 1] - 7000) / 10.)
                if (strobedEvents[i] == LookupTable['TARG1_X']):
                    if i == (nStrobedEvents - 1):
                        self.waitingForTarg1X = True
                    else:
                        self.stimInfo['targ1x'].append(
                            (strobedEvents[i + 1] - 7500) / 10.)
                if (strobedEvents[i] == LookupTable['TARG2_X']):
                    if i == (nStrobedEvents - 1):
                        self.waitingForTarg2X = True
                    else:
                        self.stimInfo['targ2x'].append(
                            (strobedEvents[i + 1] - 7500) / 10.)
                if (strobedEvents[i] == LookupTable['TASKID']):
                    if i == (nStrobedEvents - 1):
                        self.waitingForTaskID = True
                    else:
                        self.stimInfo['taskid'].append(
                            strobedEvents[i + 1] - 7000)

                # relevent trial timestamps
                if (strobedEvents[i] == LookupTable['FIX_ON']):
                    self.stimInfo['t_fixon'].append(strobedEventTimes[i])
                    self.got_t_fixon = True
                if (strobedEvents[i] == LookupTable['TARGETS_ON']):
                    self.stimInfo['t_targetson'].append(strobedEventTimes[i])
                    self.got_t_targetson = True
                if (strobedEvents[i] == LookupTable['DOTS_ON']):
                    self.stimInfo['t_dotson'].append(strobedEventTimes[i])
                    self.got_t_dotson = True
                if (strobedEvents[i] == LookupTable['DOTS_OFF']):
                    self.stimInfo['t_dotsoff'].append(strobedEventTimes[i])
                    self.got_t_dotsoff = True
                if (strobedEvents[i] == LookupTable['FIX_OFF']):
                    self.stimInfo['t_fixoff'].append(strobedEventTimes[i])
                    self.got_t_fixoff = True
                if (strobedEvents[i] == LookupTable['SACCADE_DETECTED']):
                    self.stimInfo['t_saccdet'].append(strobedEventTimes[i])
                    self.stimInfo['t_response'].append(strobedEventTimes[i])
                    self.got_t_response = True
                if (strobedEvents[i] == LookupTable['REACH_DETECTED']):
                    self.stimInfo['t_reachdet'].append(strobedEventTimes[i])
                    self.stimInfo['t_response'].append(strobedEventTimes[i])
                    self.got_t_response = True

                # response
                if (strobedEvents[i] == LookupTable['WRONG']):
                    self.stimInfo['response'].append(0)
                    self.gotResponse = True
                    self.atleast_onevalid = True
                if (strobedEvents[i] == LookupTable['CORRECT']):
                    self.stimInfo['response'].append(1)
                    self.gotResponse = True
                    self.atleast_onevalid = True
                if (strobedEvents[i] == LookupTable['NOCHOICE']):
                    self.stimInfo['response'].append(-1)
                    self.gotResponse = True
                if (strobedEvents[i] == LookupTable['FIXBREAK']):
                    self.stimInfo['response'].append(-2)
                    self.gotResponse = True
                if (strobedEvents[i] == LookupTable['FIXHANDBREAK']):
                    self.stimInfo['response'].append(-3)
                    self.gotResponse = True

                if (strobedEvents[i] == LookupTable['TRIALEND']):
                    if not(self.gotResponse):
                        self.stimInfo['response'].append(-4)

                    # this ensures that each array of timestamps
                    # is the same length;
                    # if we did not actually receive the timestamps,
                    # a blank entry is created;
                    # this then makes it easier to use boolean selectors
                    # later on;
                    potential_missing_timestamps = [
                        't_fixon', 't_fixoff', 't_dotson', 't_dotsoff',
                        't_targetson', 't_response']
                    for j in potential_missing_timestamps:
                        if not(eval('self.got_' + j)):
                            eval('self.stimInfo[\'' + j + '\'].append([])')

                    self.stimInfo['t_end'].append(strobedEventTimes[i])

                    if not(self.got_dotdirection):
                        self.stimInfo['direction'].append([])

                    # Put the spks into this "trial" ordered dictionary,
                    # making use of the trial begin and trial end timestamps
                    timeSel = np.logical_and(
                        np.array(self.tempSpkHolder['TimeStamps']) >
                        self.stimInfo['t_begin'][-1],
                        np.array(self.tempSpkHolder['TimeStamps']) <
                        self.stimInfo['t_end'][-1])
                    self.stimInfo['spk_times'].append(
                        np.array(self.tempSpkHolder['TimeStamps'])[timeSel])
                    self.stimInfo['spk_chans'].append(
                        np.array(self.tempSpkHolder['Channels'])[timeSel])
                    self.stimInfo['spk_units'].append(
                        np.array(self.tempSpkHolder['Units'])[timeSel])

                    # Clear the temporary spike holder so we're not using
                    # too much memory;
                    # I don't think this is really necessary but whatevs
                    self.tempSpkHolder = defaultdict(list)

                    # Reinitialize flags to help us deal with the
                    # rolling information
                    self.trialParsed = True
                    self.inTrial = False
                    self.gotResponse = False

                    self.got_t_fixon = False
                    self.got_t_fixoff = False
                    self.got_t_targetson = False
                    self.got_t_dotson = False
                    self.got_t_dotsoff = False
                    self.got_t_response = False

                    self.got_dotdirection = False

    def getSortedSpkTimes(self):
        response_status = np.array(self.stimInfo['response'])

        # get valid/completed trials only; this function should
        # only get called once
        # at least one valid trial has been received;
        # otherwise, some of the code below would break
        boolsel = response_status > 0

        t_begin = np.array(self.stimInfo['t_begin'], dtype=object)[boolsel]
        t_fixon = np.array(self.stimInfo['t_fixon'], dtype=object)[boolsel]
        t_fixoff = np.array(self.stimInfo['t_fixoff'], dtype=object)[boolsel]
        t_targetson = np.array(
            self.stimInfo['t_targetson'], dtype=object)[boolsel]
        t_dotson = np.array(self.stimInfo['t_dotson'], dtype=object)[boolsel]
        t_response = np.array(
            self.stimInfo['t_response'], dtype=object)[boolsel]
        t_end = np.array(self.stimInfo['t_end'], dtype=object)[boolsel]
        taskid = np.array(self.stimInfo['taskid'], dtype=object)[boolsel]

        t_mixture = np.zeros(np.sum(boolsel))
        t_mixture[taskid < 20] = np.atleast_1d(
            np.float64(t_fixoff[taskid < 20]))
        t_mixture[taskid >= 20] = np.atleast_1d(
            np.float64(t_dotson[taskid >= 20]))

        side_targ = np.sign(np.float64(np.array(
            self.stimInfo['targ1x'], dtype=object)[boolsel][taskid < 20]))
        side_direction = np.sign(np.cos(np.float64(np.array(
            self.stimInfo['direction'], dtype=object)[boolsel][taskid>=20]) * np.pi / 180))
        side = np.zeros(np.sum(boolsel))
        side[taskid < 20] = np.atleast_1d(side_targ)
        side[taskid >= 20] = np.atleast_1d(side_direction)

        spk_times = list()
        spk_chans = list()
        spk_units = list()
        ctr = 0
        for i in boolsel:
            if i:
                spk_times.append(self.stimInfo['spk_times'][ctr])
                spk_chans.append(self.stimInfo['spk_chans'][ctr])
                spk_units.append(self.stimInfo['spk_units'][ctr])
            ctr += 1

        return {'ntrials': len(side), 'side': side,
                't_begin': t_begin, 't_end': t_end,
                't_fixon': t_fixon, 't_fixoff': t_fixoff,
                't_targetson': t_targetson, 't_dotson': t_dotson,
                't_response': t_response,
                't_mixture': t_mixture,
                'spk_times': spk_times, 'spk_chans': spk_chans,
                'spk_units': spk_units}

    def resetData(self):
        self.stimInfo = defaultdict(list)

        self.inTrial = False
        self.gotResponse = False
        self.trialParsed = False
        self.atleast_onevalid = False

        # when left hanging from a single query
        self.waitingForDir = False
        self.waitingForCoh = False
        self.waitingForTarg1X = False
        self.waitingForTarg2X = False
        self.waitingForTaskID = False


class RastersPlotter:
    def __init__(self, root, figure, args):

        (spikeNumber, nTrials, psthBinwidth,
         xmin0, xmax0, xmin1, xmax1, xmin2, xmax2) = args
        self.spikeNumberToPlot = spikeNumber
        self.nTrialsToPlot = nTrials
        self.psthBinwidth = psthBinwidth
        self.xmin0 = xmin0
        self.xmax0 = xmax0
        self.xmin1 = xmin1
        self.xmax1 = xmax1
        self.xmin2 = xmin2
        self.xmax2 = xmax2

        self.Data = DataGatherer()

        self.root = root
        self.figure = figure
        self.initializePlots()
        self.BACKGROUND_UPDATE_REQUIRED = True

    def initializePlots(self):
        self.axes = list()
        self.canvases = list()
        self.raster_lines = list()
        self.trial_lines = list() # to mark with a horizontal tick individual trials
        self.psth_left_lines = list()
        self.psth_right_lines = list()

        for i in range(NPLOTS):
            self.axes.append(self.figure.add_subplot(3, 3, i + 1))
            self.canvases.append(self.axes[i].figure.canvas)

            xmin = eval('self.xmin' + repr(int(i / 3)) + '.get()')
            xmax = eval('self.xmax' + repr(int(i / 3)) + '.get()')
            self.axes[i].set_xlim(xmin, xmax)
            self.axes[i].axvline(0, color='k')

            # Set up the labels/ticks/titles

            if i % 3 == 2: # for the PSTHs, the right column of plots
                self.axes[i].set_ylim(0, 200)
                self.axes[i].yaxis.set_ticks_position('both')
                self.axes[i].yaxis.set_label_position('right')
                self.axes[i].tick_params('y', labelleft=False, labelright=True)
                self.axes[i].set_ylabel('Firing Rate (Hz)')
            else:
                self.axes[i].set_ylim(0, 1)
                self.axes[i].yaxis.set_ticks_position('none')
                self.axes[i].yaxis.set_major_formatter(NullFormatter())

            if int(i / 3) == 2:
                self.axes[i].set_xlabel('Time (ms)')
            else:
                self.axes[i].set_xlabel('')

            if (i % 3 == 0) & (int(i / 3) == 0):
                self.axes[i].set_title('LEFT', size='medium')
            if (i % 3 == 1) & (int(i / 3) == 0):
                self.axes[i].set_title('RIGHT', size='medium')
            if (i % 3 == 2) & (int(i / 3) == 0):
                self.axes[i].set_title('PSTH', size='medium')

            if (i % 3 == 0) & (int(i / 3) == 0):
                self.axes[i].set_ylabel('TARGON', rotation=30)
            if (i % 3 == 0) & (int(i / 3) == 1):
                self.axes[i].set_ylabel('FIXOFF/  \nDOTSON', rotation=30)
            if (i % 3 == 0) & (int(i / 3) == 2):
                self.axes[i].set_ylabel('RESPONSE', rotation=30)

            self.axes[i].tick_params('both', labelsize='xx-small')

            self.axes[i].figure.canvas.draw()

            # NOT ALL LINES WILL BE USED IN ALL PLOTS!!!
            self.raster_lines.append(Line2D(
                [], [], animated=True, alpha=1.0, marker='|', markersize=10,
                markeredgewidth=1, color=COLORS[i % 3]))
            self.axes[i].add_line(self.raster_lines[i])

            self.trial_lines.append(Line2D(
                [], [], animated=True, alpha=1.0, marker='>', markersize=5,
                color='blue'))
            self.axes[i].add_line(self.trial_lines[i])

            self.psth_left_lines.append(Line2D(
                [], [], animated=True, color='green'))
            self.axes[i].add_line(self.psth_left_lines[i])
            self.psth_right_lines.append(Line2D(
                [], [], animated=True, color='magenta'))
            self.axes[i].add_line(self.psth_right_lines[i])

            self.canvases[i].mpl_connect('draw_event', self.updateBackground)

    def updateBackground(self, event):
        self.BACKGROUND_UPDATE_REQUIRED = True

    def updateDataToPlot(self):
        spkDict = self.Data.getSortedSpkTimes()
        spikeNumber = self.spikeNumberToPlot.get()

        # convert to Plexon numbers
        channelNumber = int(spikeNumber / 5) + 1
        unitNumber = spikeNumber % 5

        self.timestamps = list()

        for i in range(spkDict['ntrials']):
            self.timestamps.append(spkDict['spk_times'][i][np.logical_and(
                spkDict['spk_chans'][i] == channelNumber,
                spkDict['spk_units'][i] == unitNumber)])

        self.side = spkDict['side']

        # the event times on which we will subsequently align the timestamps
        self.t_fixon = spkDict['t_fixon']
        self.t_fixoff = spkDict['t_fixoff']
        self.t_targetson = spkDict['t_targetson']
        self.t_dotson = spkDict['t_dotson']
        self.t_response = spkDict['t_response']
        self.t_mixture = spkDict['t_mixture']

    def updatePlots(self):
        # Save the empty backgrounds for quicker refreshes
        # I had to do it here instead of in __init__
        if self.BACKGROUND_UPDATE_REQUIRED:
            self.initializePlots()
            self.backgrounds = list()
            for i in range(NPLOTS):
                self.backgrounds.append(
                    self.canvases[i].copy_from_bbox(self.axes[i].bbox))
            self.BACKGROUND_UPDATE_REQUIRED = 0

        # Restore the empty backgrounds
        for i in range(NPLOTS):
            self.canvases[i].restore_region(self.backgrounds[i])

        for i in range(NPLOTS):
            # the different rows of plots have different align events
            if (i == 0) | (i == 1) | (i == 2):
                t_align = self.t_targetson
            elif (i == 3) | (i == 4) | (i == 5):
                t_align = self.t_mixture
            elif (i == 6) | (i == 7) | (i == 8):
                t_align = self.t_response

            timestamps = list()
            if i % 3 == 1:
                timestamps = [
                    self.timestamps[j]
                    for j in range(len(self.side)) if self.side[j] == 1.]
                t_align = t_align[self.side == 1.]
            elif i % 3 == 0:
                timestamps = [
                    self.timestamps[j]
                    for j in range(len(self.side)) if self.side[j] == -1.]
                t_align = t_align[self.side == -1.]

            if i % 3 < 2: # means it's a raster plot

                pixels_per_yunit = (
                    self.axes[i].transData.transform_point((1, 1)) -
                    self.axes[i].transData.transform_point((0, 0)))[1]

                # How many total spikes are we dealing with?
                # Get the most recent n_trials
                n_trials = np.min([self.nTrialsToPlot.get(), len(timestamps)])
                timestamps = timestamps[-1 * n_trials:]
                t_align = t_align[-1 * n_trials:]
                n_total_spikes = np.sum([len(ts) for ts in timestamps])

                # Pre-allocate... we're going to make one long array of
                # all spikes and then we will vertically offset the
                # individual trials
                xdataRasters = np.zeros(n_total_spikes)
                ydataRasters = np.zeros(n_total_spikes)

                xdataTrialmarkers = np.zeros(n_trials)
                ydataTrialmarkers = np.zeros(n_trials)

                offset = 0
                for j in range(n_trials):
                    xdataRasters[offset:(offset + len(timestamps[j]))] = \
                        timestamps[j] - t_align[j]
                    ydataRasters[offset:(offset + len(timestamps[j]))] = \
                        np.repeat(
                            self.axes[i].get_ylim()[1] - .025  - j * np.min([1 / n_trials, .1]), len(timestamps[j]))

                    xdataTrialmarkers[j] =  self.axes[i].get_xlim()[0] + 10
                    ydataTrialmarkers[j] =  self.axes[i].get_ylim()[1] - .025  - j * np.min([1/n_trials,.1])

                    offset += len(timestamps[j])

                xmin = eval('self.xmin' + repr(int(i / 3)) + '.get()')
                xmax = eval('self.xmax' + repr(int(i / 3)) + '.get()')
                psthBinwidth = self.psthBinwidth.get()
                bincounts, binedges = np.histogram(
                    xdataRasters, np.arange(xmin, xmax, 1))
                if n_trials > 0:
                    psth = movavg(bincounts / n_trials * 1000, psthBinwidth)
                else:
                    psth = movavg(bincounts * 1000, psthBinwidth)
                bincenters = movavg(binedges, psthBinwidth)[0:-1]

                # the following piece of code leaves behind the psths
                # to be pick up by plots 3, 6, and 9
                if i % 3 == 1:
                    self.psth_right = psth
                    self.bincenters_right = bincenters
                elif i % 3 == 0:
                    self.psth_left = psth
                    self.bincenters_left = bincenters

                # update the actual plots
                self.raster_lines[i].set_data(xdataRasters, ydataRasters)
                self.raster_lines[i].set_linestyle('None')
                self.raster_lines[i].set_markersize(
                    np.min([pixels_per_yunit / n_trials, 10]))
                self.axes[i].draw_artist(self.raster_lines[i])

                self.trial_lines[i].set_data(
                    xdataTrialmarkers, ydataTrialmarkers)
                self.trial_lines[i].set_linestyle('None')
                self.raster_lines[i].set_markersize(
                    np.min([pixels_per_yunit / n_trials, 5]))
                self.axes[i].draw_artist(self.trial_lines[i])

                # blit that shit, whatever it does
                self.canvases[i].blit(self.axes[i].bbox)
            else:
                # means we're doing a histogram plot
                # here's where we pick up those self.psth and self.bincenters
                # that we left behind
                self.psth_left_lines[i].set_data(
                    self.bincenters_left, self.psth_left)
                self.axes[i].draw_artist(self.psth_left_lines[i])

                self.psth_right_lines[i].set_data(
                    self.bincenters_right, self.psth_right)
                self.axes[i].draw_artist(self.psth_right_lines[i])

                maxFR = np.max([10, np.max(np.concatenate(
                    [self.psth_left, self.psth_right])) * 1.25])

                # self.axes[i].xaxis.set_major_formatter(NullFormatter())
                # self.axes[i].yaxis.set_major_formatter(NullFormatter())

                # blit that shit, whatever it does
                self.canvases[i].blit(self.axes[i].bbox)

    def updateWrapper(self):
        self.updateDataToPlot()
        self.updatePlots()

    def updateXLimits(self):
        for i in range(NPLOTS):
            xmin = eval('self.xmin' + repr(int(i / 3)) + '.get()')
            xmax = eval('self.xmax' + repr(int(i / 3)) + '.get()')
            self.axes[i].set_xlim(xmin, xmax)
        self.BACKGROUND_UPDATE_REQUIRED = True

        if self.Data.atleast_onevalid:
            self.updateWrapper()

    def updateUnitToPlot(self):
        if self.Data.atleast_onevalid:
            self.updateWrapper()

    def DataGatherLoop(self):
        # get the data from the MAP server
        self.Data.getMAPEvents()

        # if a whole trial's been gotten, update plots
        if self.Data.trialParsed & self.Data.atleast_onevalid:
            self.updateWrapper()

        # important, polling, don't go too fast
        self.callbackID = self.canvases[0].get_tk_widget().after(
            250, self.DataGatherLoop)

    def reset(self):
        self.canvases[0].get_tk_widget().after_cancel(self.callbackID)
        self.BACKGROUND_UPDATE_REQUIRED = True
        self.Data.resetData()
        self.updateWrapper()
        self.DataGatherLoop()

    def quit(self):
        self.Data.plexdll.PL_CloseClient()
        self.canvases[0].get_tk_widget().after_cancel(self.callbackID)
        self.root.destroy()
        self.root.quit()


root = Tk.Tk()
root.wm_title("Rasters/SDFs split by side")

f = Figure()

canvas = FigureCanvasTkAgg(f, master=root)
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
canvas.show()

spikeNumber = Tk.IntVar()
spikeNumber.set(1)

n_trials_to_plot = Tk.IntVar()
n_trials_to_plot.set(20)

psth_binwidth = Tk.IntVar()
psth_binwidth.set(50)

xmin0 = Tk.IntVar()
xmin0.set(-500)
xmax0 = Tk.IntVar()
xmax0.set(500)

xmin1 = Tk.IntVar()
xmin1.set(-500)
xmax1 = Tk.IntVar()
xmax1.set(750)

xmin2 = Tk.IntVar()
xmin2.set(-750)
xmax2 = Tk.IntVar()
xmax2.set(500)

Rasters = RastersPlotter(root, f, [spikeNumber, n_trials_to_plot,
                                   psth_binwidth, xmin0, xmax0,
                                   xmin1, xmax1, xmin2, xmax2])

SpikeFrame = Tk.LabelFrame(
    master=root, text='SPIKE NUMBER TO PLOT', borderwidth=5)
SpikeFrame.pack(
    padx=10, pady=10, side=Tk.LEFT, anchor='n', fill='both')

UNITS1 = [("Unsorted", 0),
          ("Unit 1", 1),
          ("Unit 2", 2),
          ("Unit 3", 3),
          ("Unit 4", 4)]
UNITS2 = [("Unsorted", 5),
          ("Unit 1", 6),
          ("Unit 2", 7),
          ("Unit 3", 8),
          ("Unit 4", 9)]
UNITS3 = [("Unsorted", 10),
          ("Unit 1", 11),
          ("Unit 2", 12),
          ("Unit 3", 13),
          ("Unit 4", 14)]
UNITS4 = [("Unsorted", 15),
          ("Unit 1", 16),
          ("Unit 2", 17),
          ("Unit 3", 18),
          ("Unit 4", 19)]
UNITS5 = [("Unsorted", 20),
          ("Unit 1", 21),
          ("Unit 2", 22),
          ("Unit 3", 23),
          ("Unit 4", 24)]

UnitFrame1 = Tk.LabelFrame(master=SpikeFrame, text='Channel 1')
UnitFrame1.pack(side=Tk.LEFT, padx=5, pady=10, anchor='w')
for text, value in UNITS1:
    b = Tk.Radiobutton(master=UnitFrame1, text=text,
                       variable=spikeNumber, value=value,
                       command=Rasters.updateUnitToPlot)
    b.pack(anchor='w')

UnitFrame2 = Tk.LabelFrame(master=SpikeFrame, text='Channel 2')
UnitFrame2.pack(side=Tk.LEFT, padx=5, pady=10, anchor='w')
for text, value in UNITS2:
    b = Tk.Radiobutton(master=UnitFrame2, text=text,
                       variable=spikeNumber, value=value,
                       command=Rasters.updateUnitToPlot)
    b.pack(anchor='w')

UnitFrame3 = Tk.LabelFrame(master=SpikeFrame, text='Channel 3')
UnitFrame3.pack(side=Tk.LEFT, padx=5, pady=10, anchor='w')
for text, value in UNITS3:
    b = Tk.Radiobutton(master=UnitFrame3, text=text,
                       variable=spikeNumber, value=value,
                       command=Rasters.updateUnitToPlot)
    b.pack(anchor='w')

UnitFrame4 = Tk.LabelFrame(master=SpikeFrame, text='Channel 4')
UnitFrame4.pack(side=Tk.LEFT, padx=5, pady=10, anchor='w')
for text, value in UNITS4:
    b = Tk.Radiobutton(master=UnitFrame4, text=text,
                       variable=spikeNumber, value=value,
                       command=Rasters.updateUnitToPlot)
    b.pack(anchor='w')

UnitFrame5 = Tk.LabelFrame(master=SpikeFrame, text='Channel 5')
UnitFrame5.pack(side=Tk.LEFT, padx=5, pady=10, anchor='w')
for text, value in UNITS5:
    b = Tk.Radiobutton(master=UnitFrame5, text=text,
                       variable=spikeNumber, value=value,
                       command=Rasters.updateUnitToPlot)
    b.pack(anchor='w')

################
#### OPTIONS ###
################
OptionsFrame = Tk.LabelFrame(master=root, text='PLOT LIMITS',
                             borderwidth=5)
OptionsFrame.pack(padx=10, pady=10, side=Tk.LEFT, anchor='w', fill='both')

x_min = [xmin0, xmin1, xmin2]
x_max = [xmax0, xmax1, xmax2]
labels = ['X Limits for Top Row', 'X Limits for Middle Row',
          'X Limits for Bottom Row']
for i in range(3):
    XLimFrame = Tk.LabelFrame(master=OptionsFrame, text=labels[i])
    XLimFrame.pack(padx=5, pady=10, side=Tk.TOP, anchor='n')
    XMinLabel = Tk.Label(master=XLimFrame, text='Pre-time')
    XMinLabel.pack(side=Tk.LEFT, anchor='e')
    XMinSpinbox = Tk.Spinbox(master=XLimFrame, textvariable=x_min[i],
                             command=Rasters.updateXLimits, from_=-2000,
                             to=0, increment=50, width=5)
    XMinSpinbox.pack(side=Tk.LEFT, anchor='e', padx=2)
    XMaxLabel = Tk.Label(master=XLimFrame, text='Post-time')
    XMaxLabel.pack(side=Tk.LEFT, anchor='e')
    XMaxSpinbox = Tk.Spinbox(master=XLimFrame, textvariable=x_max[i],
                             command=Rasters.updateXLimits,
                             from_=0, to=2000, increment=50, width=5)
    XMaxSpinbox.pack(side=Tk.LEFT, anchor='e', padx=2)


#####################################################
# NTrials, Binwidth and Button all in same container#
#####################################################
ContainerFrame = Tk.Frame(master=root)
ContainerFrame.pack(padx=10, pady=10, side=Tk.LEFT, anchor='n', fill='both')

################
#### NTRIALS ###
################

NTrialsFrame = Tk.LabelFrame(master=ContainerFrame, text='# TRIALS',
                             borderwidth=5)
NTrialsFrame.pack(side=Tk.TOP, anchor='w', fill='both')
NTrialsSpinbox = Tk.Spinbox(master=NTrialsFrame, textvariable=n_trials_to_plot,
                            command=Rasters.updateWrapper, from_=10,
                            to=200, increment=10, width=5)
NTrialsSpinbox.pack(side=Tk.LEFT, anchor='e', padx=10, pady=10)


#################
#### BINWIDTH ###
#################

BinwidthFrame = Tk.LabelFrame(master=ContainerFrame, text='PSTH BINWDITH',
                              borderwidth=5)
BinwidthFrame.pack(side=Tk.TOP, anchor='w', fill='both')
BinwidthSpinbox = Tk.Spinbox(master=BinwidthFrame, textvariable=psth_binwidth,
                             command=Rasters.updateWrapper, from_=5, to=200,
                             increment=5, width=5)
BinwidthSpinbox.pack(side=Tk.LEFT, anchor='e', padx=10, pady=10)

################
#### BUTTONS ###
################

ButtonFrame = Tk.LabelFrame(master=ContainerFrame, text='ACTIONS',
                            borderwidth=5)
ButtonFrame.pack(side=Tk.TOP, anchor='w', fill='both')
button = Tk.Button(master=ButtonFrame, text='Quit', command=Rasters.quit,
                   fg='red', pady=5)
button.pack(side=Tk.BOTTOM, fill='x', anchor='e', expand=False, padx=10)
button = Tk.Button(master=ButtonFrame, text='Reset', command=Rasters.reset,
                   fg='blue', pady=5)
button.pack(side=Tk.BOTTOM, fill='x', anchor='e', expand=False, padx=10)

Rasters.DataGatherLoop()
root.mainloop()
