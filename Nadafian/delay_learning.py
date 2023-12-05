#!/bin/python
import sys, os
from typing import Any
sys.path.append('/home/amelie/Documents/doctorat/scripts/EvData/read_event_data')
sys.path.append('/home/amelie/Documents/doctorat/scripts/EvData/translate_2_formats')

import itertools as it
from pyNN.utility import get_simulator, init_logging, normalized_filename
from quantities import ms
from random import randint
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import imageio
from datetime import datetime as dt
import neo
import numpy as np
from events2spikes import ev2spikes

start = dt.now()

sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file", {"action": "store_true"}),
                             ("--metrics", "Display metrics computed from simulation run", {"action": "store_true"}),
                             ("--nb-convolution", "The number of convolution layers of the model", {"action": "store", "type": int, 'default': 2}),
                             ("--t", "Length of the simulation (in microseconds)", {"action": "store", "type": float, 'default': 1e6}),
                             ("--noise", "Run on noisy data", {"action": "store_true"}),
                             ("--verbose", "Display all information", {"action": "store_true"}),
                             ("--save", "Save information in output file", {"action": "store_true"}),
                             ("--debug", "Print debugging information", {"action": "store_true"}))

if options.debug:
    os.makedirs('./logs', exist_ok=True)
    init_logging('logs/'+dt.now().strftime("%Y%m%d-%H%M%S")+'.txt', debug=True)

if sim == "nest":
    from pyNN.nest import *

sim.setup(timestep=0.01)

### Parameters

if options.noise: 
    parameters = {
        'Rtarget':  0.0025,     # Target neural activation rate
        'lambda_w': 0.00002,    # Homeostasis application rate for weights
        'lambda_d': 0.0001,     # Homeostasis application rate for delays
        'STDP_w':   0.01,       # STDP increment/decrement range for weights
        'STDP_d':   1.0         # STDP increment/decrement range for delays
    }
else:
    parameters = {
        'Rtarget':  0.003,      # Target neural activation rate
        'lambda_w': 0.00003,    # Homeostasis application rate for weights
        'lambda_d': 0.0006,     # Homeostasis application rate for delays
        'STDP_w':   0.01,       # STDP increment/decrement range for weights
        'STDP_d':   1.0         # STDP increment/decrement range for delays
    }

OUTPUT_PATH_GENERIC = "./results"
time_now = dt.now().strftime("%Y%m%d-%H%M%S")
results_path = os.path.join(OUTPUT_PATH_GENERIC, time_now)

### Directions

DIRECTIONS = {
    -1: "INDETERMINATE", 
    0: "SOUTH-EAST ↘︎", 
    1: "SOUTH-WEST ↙︎", 
    2: "NORTH-WEST ↖︎", 
    3: "NORTH-EAST ↗︎", 
    4: "EAST →", 
    5: "SOUTH ↓", 
    6: "WEST ←", 
    7: "NORTH ↑"
} # KEY=DIRECTIONS ID ; VALUE=STRING REPRESENTING THE DIRECTION

NB_CONV_LAYERS = options.nb_convolution
if NB_CONV_LAYERS < 2 or NB_CONV_LAYERS > 8:
    sys.exit("[Error] The number of convolution layers should be at least 2. The current implementation allows for a maximum number of layers of 4.")

NB_DIRECTIONS = min(len(DIRECTIONS)-1, NB_CONV_LAYERS) # No more than available directions, and at least 2 directions. -1 to ignore INDETERMINATE

LEARNING = False
learning_time = 'NA'

### Generate input data

time_data = int(options.t)
temporal_reduction = 1e3
pattern_interval = 1e2
pattern_duration = 5
num = time_data//pattern_interval

# The input should be at least 13*13 for a duration of 5 since we want to leave a margin of 4 neurons on the edges when generating data
x_input = 13
filter_x = 5
x_output = x_input - filter_x + 1

y_input = 13
filter_y = 5
y_output = y_input - filter_y + 1

x_margin = y_margin = 4

# Dataset Generation
input_data = {} # key: (begin motion, end motion) - value: id direction
input_events = np.zeros((0,4))
for t in range(int(time_data/pattern_interval)):
    
    direction = randint(0,NB_DIRECTIONS - 1)  # NB_DIRECTIONS possible directions
    input_data[(
        t * pattern_interval, 
        t*pattern_interval + pattern_duration - 1
    )] = direction

    # SOUTH-EAST ↘︎
    if direction==0:
        start_x = randint(x_margin, x_input-pattern_duration-x_margin) # We leave a margin of 4 neurons on the edges of the input layer so that the whole movement can be seen by the convolution window
        start_y = randint(y_margin, y_input-pattern_duration-y_margin)
        input_events = np.concatenate((
            input_events,
            [[start_x+d, start_y+d, 1, d+t*pattern_interval] for d in range(pattern_duration)]
        ), axis=0)
    
    # SOUTH-WEST ↙︎
    elif direction==1:
        start_x = randint(x_input-x_margin-1, x_input-pattern_duration) # We leave a margin of 4 neurons on the edges of the input layer so that the whole movement can be seen by the convolution window
        start_y = randint(y_margin, y_input-pattern_duration-y_margin)
        input_events = np.concatenate((
            input_events,
            [[start_x-d, start_y+d, 1, d+t*pattern_interval] for d in range(pattern_duration)]
        ), axis=0)
    
    # NORTH-WEST ↖︎
    elif direction == 2:
        start_x = randint(x_input-x_margin-1, x_input-pattern_duration) # We leave a margin of 4 neurons on the edges of the input layer so that the whole movement can be seen by the convolution window
        start_y = randint(y_input-y_margin-1, y_input-pattern_duration)
        input_events = np.concatenate((
            input_events, 
            [[start_x-d, start_y-d, 1, d+t*pattern_interval] for d in range(pattern_duration)]
        ), axis=0)

    # NORTH-EAST ↗︎
    elif direction == 3:
        start_x = randint(x_margin, x_input-pattern_duration-x_margin) # We leave a margin of 4 neurons on the edges of the input layer so that the whole movement can be seen by the convolution window
        start_y = randint(y_input-y_margin-1, y_input-pattern_duration)
        input_events = np.concatenate((
            input_events, 
            [[start_x+d, start_y-d, 1, d+t*pattern_interval] for d in range(pattern_duration)]
        ), axis=0)
    
    # EAST →
    elif direction == 4:
        start_x = randint(x_margin, x_input-pattern_duration-x_margin)
        start_y = randint((y_margin + (y_input-y_margin)) // 2, (y_margin + (y_input-y_margin)) // 2)
        input_events = np.concatenate((
            input_events, [[start_x+d, start_y, 1, d+t*pattern_interval] for d in range(pattern_duration)]), 
            axis=0
        )
    
    # SOUTH ↓
    elif direction == 5:
        start_x = randint((x_margin + (x_input-x_margin)) // 2, (x_margin + (x_input-x_margin)) // 2)
        start_y = randint(y_margin, y_input-pattern_duration-y_margin)
        input_events = np.concatenate((
            input_events, [[start_x, start_y+d, 1, d+t*pattern_interval] for d in range(pattern_duration)]), 
            axis=0
        )

    # WEST ←
    elif direction == 6:
        start_x = randint(x_input-x_margin-1, x_input-pattern_duration)
        start_y = randint((y_margin + (y_input-y_margin)) // 2, (y_margin + (y_input-y_margin)) // 2)
        input_events = np.concatenate((
            input_events, [[start_x-d, start_y, 1, d+t*pattern_interval] for d in range(pattern_duration)]), 
            axis=0
        )

    # NORTH ↑
    elif direction == 7:
        start_x = randint((x_margin + (x_input-x_margin)) // 2, (x_margin + (x_input-x_margin)) // 2)
        start_y = randint(y_input-y_margin-1, y_input-pattern_duration)
        input_events = np.concatenate((
            input_events, [[start_x, start_y-d, 1, d+t*pattern_interval] for d in range(pattern_duration)]), 
            axis=0
        )

input_spiketrain, _, _ = ev2spikes(input_events, width=x_input, height=y_input)


### Build Network 

# Populations

Input = sim.Population(
    x_input*y_input,  
    sim.SpikeSourceArray(spike_times=input_spiketrain), 
    label="Input"
)
Input.record("spikes")

# 'tau_m': 20.0,       # membrane time constant (in ms)   
# 'tau_refrac': 30.0,  # duration of refractory period (in ms) 0.1 de base
# 'v_reset': -70.0,    # reset potential after a spike (in mV) 
# 'v_rest': -70.0,     # resting membrane potential (in mV)
# 'v_thresh': -5.0,    # spike threshold (in mV) -5 de base
Convolutions_parameters = {
    'tau_m': 10.0,       # membrane time constant (in ms)   
    'tau_refrac': 10.0,  # duration of refractory period (in ms) 0.1 de base
    'v_reset': -70.0,    # reset potential after a spike (in mV) 
    'v_rest': -70.0,     # resting membrane potential (in mV)
    'v_thresh': -5.0,    # spike threshold (in mV) -5 de base
}

# The size of a convolution layer with a filter of size x*y is input_x-x+1 * input_y-y+1 
convolutions = []
for i in range(NB_CONV_LAYERS):
    Conv_i = sim.Population(
        x_output*y_output, 
        sim.IF_cond_exp(**Convolutions_parameters),
        label="Convolution "+str(i+1)
    )
    Conv_i.record('spikes')
    convolutions.append(Conv_i)


# List connector

# weight_N = 0.35 
# delays_N = 15.0 
# weight_teta = 0.005 
# delays_teta = 0.05 
weight_N = 0.5
delays_N = 15.0 
weight_teta = 0.01 
delays_teta = 0.02 

weight_conv = np.random.normal(weight_N, weight_teta, size=(NB_CONV_LAYERS, filter_x, filter_y))
delay_conv =  np.random.normal(delays_N, delays_teta, size=(NB_CONV_LAYERS, filter_x, filter_y))

input2conv_conn = [[] for _ in range(NB_CONV_LAYERS)]
c = 0

for in2conv_conn in input2conv_conn:

    for X,Y in it.product(range(x_output), range(y_output)):

        idx_conv = np.ravel_multi_index( (X,Y) , (x_output, y_output) )

        conn = []
        for x, y in it.product(range(filter_x), range(filter_y)):
            w = weight_conv[c, x, y]
            d = delay_conv[ c, x, y]
            idx_in = np.ravel_multi_index( (X+x,Y+y) , (x_input, y_input) )
            conn.append( ( idx_in, idx_conv, w, d ) )

        in2conv_conn += conn
    
    c += 1

# Projections - input to convolution

input2conv = []
for idx in range(NB_CONV_LAYERS):
    input2conv_i = sim.Projection(
        Input, convolutions[idx],
        connector = sim.FromListConnector(input2conv_conn[idx], column_names = ["weight", "delay"]),
        synapse_type = sim.StaticSynapse(),
        receptor_type = 'excitatory',
        label = 'Input to Convolution '+str(idx+1)
    )
    input2conv.append(input2conv_i)


# Projections - lateral inhibition between convolution

conv2conv = []
for conv_in, conv_out in it.permutations(convolutions,2):
    in2out = sim.Projection(
        conv_in, conv_out,
        connector = sim.OneToOneConnector(),
        synapse_type = sim.StaticSynapse(
            weight=50,
            delay=0.01
        ),
        receptor_type = "inhibitory",
        label = "Lateral inhibition - "+conv_in.label+" to "+conv_out.label
    )
    conv2conv.append(in2out)

# We will use this list to know which convolution layer has reached its stop condition
full_stop_condition= [False for _ in range(NB_CONV_LAYERS)]

# Each filter of each convolution layer will be put in this list and actualized at each stimulus
final_filters = [[] for _ in range(NB_CONV_LAYERS)]

# Sometimes, even with lateral inhibition, two neurons on the same location in different convolution
# layers will both spike (due to the minimum delay on those connections). So we keep track of
# which neurons in each layer has already spiked for this stimulus. (Everything is put back to False at the end of the stimulus)
neuron_activity_tag = [ 
    [
        False for _ in range((x_input-filter_x+1)*(y_input-filter_y+1))
    ]
    for _ in range(NB_CONV_LAYERS) 
]

# When a convolution layer is specialized, it is stored in this dictionary
# key: id convolution - value: id motion
motion_per_conv = {} 


### Run simulation

# Callback classes

class LastSpikeRecorder(object):

    def __init__(self, sampling_interval, pop):
        self.interval = sampling_interval
        self.population = pop
        self.global_spikes = [[] for _ in range(self.population.size)]
        self.annotations = {}
        self.final_spikes = []
        self.nb_spikes = {k: 0 for k in range(self.population.size)}
        self.nb_spikes_total = 0

        if not isinstance(self.population, list):
            self._spikes = np.ones(self.population.size) * (-1)
        else:
            self._spikes = np.ones(len(self.population)) * (-1)

    def __call__(self, t):
        if t > 0:
            if options.debug:
                print('> last spike recorder')
            
            if not isinstance(self.population, list):
                population_spikes = self.population.get_data("spikes", clear=True).segments[0].spiketrains
                self._spikes = map(
                    lambda x: x[-1].item() if len(x) > 0 else -1, 
                    population_spikes
                )
                self._spikes = np.fromiter(self._spikes, dtype=float)

                if t == self.interval:
                    for n, neuron_spikes in enumerate(population_spikes):
                        self.annotations[n] = neuron_spikes.annotations

            else:
                self._spikes = []
                for subr in self.population:
                    sp = subr.get_data("spikes", clear=True).segments[0].spiketrains
                    spikes_subr = map(
                        lambda x: x[-1].item() if len(x) > 0 else -1, 
                        sp
                    )
                    self._spikes.append(max(spikes_subr))

            assert len(self._spikes) == len(self.global_spikes)
            if len(np.unique(self._spikes)) > 1:
                idx = np.where(self._spikes != -1)[0]
                for n in idx:
                    self.global_spikes[n].append(self._spikes[n])
                    self.nb_spikes[n] += 1
                    self.nb_spikes_total += 1

        # return t+self.interval

    def get_spikes(self):
        for n, s in enumerate(self.global_spikes):
            self.final_spikes.append( neo.core.spiketrain.SpikeTrain(s*ms, t_stop=time_data, **self.annotations[n]) )
        return self.final_spikes

class WeightDelayRecorder(object):

    def __init__(self, sampling_interval, proj):
        self.interval = sampling_interval
        self.projection = proj

        self.weight = None
        self._weights = []
        self.delay = None
        self._delays = []
        self.attribute_names = self.projection.synapse_type.get_native_names('weight','delay')

    def __call__(self, t):
        if options.debug:
            print('> weight delay recorder')
        self.weight, self.delay = self.projection._get_attributes_as_arrays(self.attribute_names, multiple_synapses='sum')
        
        self._weights.append(self.weight)
        self._delays.append(self.delay)

        # return t+self.interval

    def update_weights(self, w):
        assert self._weights[-1].shape == w.shape
        self._weights[-1] = w

    def update_delays(self, d):
        assert self._delays[-1].shape == d.shape
        self._delays[-1] = d

    def get_weights(self):
        signal = neo.AnalogSignal(self._weights, units='nA', sampling_period=self.interval * ms, name="weight")
        signal.channel_index = neo.ChannelIndex(np.arange(len(self._weights[0])))
        return signal

    def get_weights(self):
        signal = neo.AnalogSignal(self._delays, units='ms', sampling_period=self.interval * ms, name="delay")
        signal.channel_index = neo.ChannelIndex(np.arange(len(self._delays[0])))
        return signal


class visualiseTime(object):
    def __init__(self, sampling_interval):
        self.interval = sampling_interval


    def __call__(self, t):
        print("Timestep : {} / {}".format(int(t), time_data))
        
        unique_full_stop_condition = np.unique(full_stop_condition)
        if len(unique_full_stop_condition) == 1 and unique_full_stop_condition[0]:
            print("!!!! FINISHED LEARNING !!!!") 
            if options.verbose:
                self.print_final_filters()
            global LEARNING, learning_time
            LEARNING = True
            learning_time = dt.now() - start
            print("complete learning time:", learning_time)

        if t > 1 and int(t) % pattern_interval==0 and options.verbose:
            self.print_final_filters()


    def print_final_filters(self):
        filter1_d, filter1_w = final_filters[0][0], final_filters[0][1] 
        filter2_d, filter2_w = final_filters[1][0], final_filters[1][1] 

        print("Delays Convolution 1 :")
        for x in filter1_d:
            for y in x:
                print("{}, ".format(y*ms), end='')
            print()
        print("Weights Convolution 1 :")
        for x in filter1_w:
            for y in x:
                print("{}, ".format(y), end='')
            print()

        print("\n")
        print("Delays Convolution 2 :")
        for x in filter2_d:
            for y in x:
                print("{}, ".format(y*ms), end='')
            print()
        print("Weights Convolution 2 :")
        for x in filter2_w:
            for y in x:
                print("{}, ".format(y), end='')
            print()


class NeuronReset(object):
    """    
    Resets neuron_activity_tag to False for all neurons in all layers.
    Also injects a negative amplitude pulse to all neurons at the end of each stimulus
    So that all membrane potentials are back to their resting values.
    """

    def __init__(self, sampling_interval, pops, t_pulse=10):
        self.interval = sampling_interval
        self.populations = pops
        self.t_pulse = t_pulse
        self.i = 0

    def __call__(self, t):
        if options.debug:
            print('> neuron reset', self.i)
        for conv in neuron_activity_tag:
            for cell in range(len(conv)):
                conv[cell] = False

        if t > 0:
            if options.verbose:
                print("!!! RESET !!!")
            if isinstance(self.populations, list):
                for pop in self.populations:
                    pulse = sim.DCSource(amplitude=-10.0, start=t, stop=t+self.t_pulse)
                    pulse.inject_into(pop)
            else:
                pulse = sim.DCSource(amplitude=-10.0, start=t, stop=t+self.t_pulse)
                pulse.inject_into(self.populations)

            self.interval = pattern_interval
        self.i += 1
        # return t + self.interval


# class InputClear(object):
#     """
#     When called, simply gets the data from the input with the 'clear' parameter set to True.
#     By periodically clearing the data from the populations the simulation goes a lot faster.
#     """

#     def __init__(self, sampling_interval, pops_to_clear_data):
#         self.interval = sampling_interval
#         self.pop_clear = pops_to_clear_data

#     def __call__(self, t):
#         if options.debug:
#             print('> input clear')
#         if t > 0:
#             print("!!! INPUT CLEAR !!!")
#             try:
#                 input_spike_train = self.pop_clear.get_data("spikes", clear=True).segments[0].spiketrains 
#             except:
#                 pass
#             self.interval = pattern_interval
        # return t + self.interval


# class LearningMechanisms(object):
#     def __init__(
#         self, 
#         sampling_interval, 
#         input_spikes_recorder, output_spikes_recorder,
#         projection, projection_delay_weight_recorder,
#         B_plus, B_minus, 
#         tau_plus, tau_minus, 
#         A_plus, A_minus, 
#         teta_plus, teta_minus, 
#         filter_d, filter_w, 
#         stop_condition, 
#         growth_factor, 
#         Rtarget=0.005, 
#         lamdad=0.002, lamdaw=0.00005, 
#         thresh_adapt=True, 
#         label=0
#     ):
#         self.interval = sampling_interval
#         self.projection = projection
#         self.input = projection.pre
#         self.output = projection.post

#         self.input_spikes = input_spikes_recorder 
#         self.output_spikes = output_spikes_recorder
#         self.DelayWeights = projection_delay_weight_recorder
        
#         # We keep the last time of spike of each neuron
#         self.input_last_spiking_times = self.input_spikes._spikes
#         self.output_last_spiking_times = self.output_spikes._spikes
        
#         self.B_plus = B_plus
#         self.B_minus = B_minus
#         self.tau_plus = tau_plus
#         self.tau_minus = tau_minus
#         self.max_delay = False # If set to False, we will find the maximum delay on first call.
#         self.filter_d = filter_d
#         self.filter_w = filter_w
#         self.A_plus = A_plus
#         self.A_minus = A_minus
#         self.teta_plus = teta_plus
#         self.teta_minus = teta_minus
#         self.c = stop_condition
#         self.growth_factor = growth_factor
#         self.label = label
#         self.thresh_adapt=thresh_adapt
        
#         # For each neuron, we count their number of spikes to compute their activation rate.
#         self.total_spike_count_per_neuron = [0 for _ in range(len(self.output))] 
        
#         # Number of times this has been called.
#         self.call_count = 0 
        
#         self.Rtarget = Rtarget
#         self.lamdaw = lamdaw 
#         self.lamdad = lamdad

#     def __call__(self, t):

#         global LEARNING
#         if options.debug:
#             print('> learning mechanisms')
#             if LEARNING:
#                 print('>> learning phase done')

#         if t == 0 :
#             print("No data")
#             # return t + pattern_interval
#         elif not LEARNING:
#             self.learn()

#     def learn(self):
#         self.call_count += 1
#         final_filters[self.label] = [self.filter_d, self.filter_w]

#         # The sum of all homeostasis delta_d and delta_t computed for each cell
#         homeo_delays_total = 0
#         homeo_weights_total = 0

#         # Since we can't increase the delays past the maximum delay set at the beginning of the simulation,
#         # we find the maximum delay during the first call
#         if self.max_delay == False:
#             self.max_delay = 0.01
#             for x in self.DelayWeights.delay:
#                 for y in x:
#                     if not np.isnan(y) and y > self.max_delay:
#                         self.max_delay = y

#         for pre_neuron in range(self.input.size):
#             if self.input_spikes._spikes[pre_neuron] != -1 and self.input_spikes._spikes[pre_neuron] > self.input_last_spiking_times[pre_neuron]:
#                 # We actualize the last time of spike for this neuron
#                 self.input_last_spiking_times[pre_neuron] = self.input_spikes._spikes[pre_neuron]
#                 if options.verbose:
#                     print("PRE SPIKE {} : {}".format(pre_neuron, self.input_spikes._spikes[pre_neuron]))


#         for post_neuron in range(self.output.size):

#             if self.output_spikes._spikes[post_neuron] != -1 and self.check_activity_tags(post_neuron):
#                 neuron_activity_tag[self.label][post_neuron] = True
#                 if options.verbose:
#                     print("***** STIMULUS {} *****".format(t//pattern_interval))

#                 self.total_spike_count_per_neuron[post_neuron] += 1

#                 # The neuron spiked during this stimulus and its threshold should be increased.
#                 # Since Nest won't allow neurons with a threshold > 0 to spike, we decrease v_rest instead.
#                 current_rest = self.output.__getitem__(post_neuron).get_parameters()['v_rest']
#                 if self.thresh_adapt:
#                     self.output.__getitem__(post_neuron).v_rest=current_rest-(1.0-self.Rtarget)
#                     self.output.__getitem__(post_neuron).v_reset=current_rest-(1.0-self.Rtarget)

#                 if options.verbose:
#                     print("=== Neuron {} from layer {} spiked ! Whith rest = {} ===".format(post_neuron, self.label, current_rest))
#                     print("Total spikes of neuron {} from layer {} : {}".format(post_neuron, self.label, self.total_spike_count_per_neuron[post_neuron]))

#                 if self.output_spikes._spikes[post_neuron] > self.output_last_spiking_times[post_neuron] and not self.stop_condition(post_neuron):
#                     # We actualize the last time of spike for this neuron
#                     self.output_last_spiking_times[post_neuron] = self.output_spikes._spikes[post_neuron]

#                     # We now compute a new delay for each of its connections using STDP
#                     for pre_neuron in range(len(self.DelayWeights.delay)):
                        
#                         # For each post synaptic neuron that has a connection with pre_neuron, we also check that both neurons
#                         # already spiked at least once.
#                         if not np.isnan(self.DelayWeights.delay[pre_neuron][post_neuron]) and not np.isnan(self.DelayWeights.weight[pre_neuron][post_neuron]) and self.input_last_spiking_times[pre_neuron] != -1 and self.output_last_spiking_times[post_neuron] != -1:

#                             # Some values here have a dimension in ms
#                             delta_t = self.output_last_spiking_times[post_neuron] - self.input_last_spiking_times[pre_neuron] - self.DelayWeights.delay[pre_neuron][post_neuron]
#                             delta_d = self.G(delta_t)
#                             delta_w = self.F(delta_t)

#                             if options.verbose:
#                                 print("STDP from layer: {} with post_neuron: {} and pre_neuron: {} deltad: {}, deltat: {}".format(self.label, post_neuron, pre_neuron, delta_d*ms, delta_t*ms))
#                                 print("TIME PRE {} : {} TIME POST 0: {} DELAY: {}".format(pre_neuron, self.input_last_spiking_times[pre_neuron], self.output_last_spiking_times[post_neuron], self.DelayWeights.delay[pre_neuron][post_neuron]))
#                             self.actualize_filter(pre_neuron, post_neuron, delta_d, delta_w)
#             else:
#                 # The neuron did not spike and its threshold should be lowered

#                 if self.thresh_adapt:
#                     current_rest = self.output.__getitem__(post_neuron).get_parameters()['v_rest']
#                     self.output.__getitem__(post_neuron).v_rest=current_rest+self.Rtarget
#                     self.output.__getitem__(post_neuron).v_reset=current_rest+self.Rtarget

#             # Homeostasis regulation per neuron
#             Robserved = self.total_spike_count_per_neuron[post_neuron]/self.call_count
#             K = (self.Rtarget - Robserved)/self.Rtarget
#             delta_d = -self.lamdad*K
#             delta_w = self.lamdaw*K
#             homeo_delays_total += delta_d  
#             homeo_weights_total += delta_w 

#             if options.verbose:
#                 print("Rate of neuron {} from layer {}: {}".format(post_neuron, self.label, Robserved))


#         if options.verbose:
#             print("****** CONVO {} homeo_delays_total: {}, homeo_weights_total: {}".format(self.label, homeo_delays_total, homeo_weights_total))
#         self.actualizeAllFilter( homeo_delays_total+self.growth_factor*self.interval, homeo_weights_total)

#         # At last we give the new delays and weights to our projections
        
#         self.DelayWeights.update_delays(self.DelayWeights.delay)
#         self.DelayWeights.update_weights(self.DelayWeights.weight)
        
#         ### HERE MODIFY 
#         self.projection.set(delay = self.DelayWeights.delay)
#         self.projection.set(weight = self.DelayWeights.weight)

#         # We update the list that tells if this layer has finished learning the delays and weights
#         full_stop_condition[self.label] = self.full_stop_check()
#         # return t + pattern_interval

#     # Computes the delay delta by applying the STDP
#     def G(self, delta_t):
#         if delta_t >= 0:
#             delta_d = -self.B_minus*np.exp(-delta_t/self.teta_minus)
#         else:
#             delta_d = self.B_plus*np.exp(delta_t/self.teta_plus)
#         return delta_d

#     # Computes the weight delta by applying the STDP
#     def F(self, delta_t):
#         if delta_t >= 0:
#             delta_w = self.A_plus*np.exp(-delta_t/self.tau_plus)
#         else:
#             delta_w = -self.A_minus*np.exp(delta_t/self.tau_minus)
#         return delta_w

#     # Given a post synaptic cell, returns if that cell has reached its stop condition for learning
#     def stop_condition(self, post_neuron):
#         for pre_neuron in range(len(self.DelayWeights.delay)):
#             if not np.isnan(self.DelayWeights.delay[pre_neuron][post_neuron]) and self.DelayWeights.delay[pre_neuron][post_neuron] <= self.c:
#                 return True
#         return False

#     # Checks if all cells have reached their stop condition
#     def full_stop_check(self):
#         for post_neuron in range(self.output.size):
#             if not self.stop_condition(post_neuron):
#                 return False
#         return True

#     # Applies the current weights and delays of the filter to all the cells sharing those
#     def actualize_filter(self, pre_neuron, post_neuron, delta_d, delta_w):
#         # We now find the delay/weight to use by looking at the filter
#         convo_coords = [post_neuron%(x_input-filter_x+1), post_neuron//(x_input-filter_x+1)]
#         input_coords = [pre_neuron%x_input, pre_neuron//x_input]
#         filter_coords = [input_coords[0]-convo_coords[0], input_coords[1]-convo_coords[1]]

#         # And we actualize delay/weight of the filter after the STDP
#         self.filter_d[filter_coords[0]][filter_coords[1]] = max(0.01, min(self.filter_d[filter_coords[0]][filter_coords[1]]+delta_d, self.max_delay))
#         self.filter_w[filter_coords[0]][filter_coords[1]] = max(0.05, self.filter_w[filter_coords[0]][filter_coords[1]]+delta_w)

#         # Finally we actualize the weights and delays of all neurons that use the same filter
#         for window_x in range(0, x_input - (filter_x-1)):
#             for window_y in range(0, y_input - (filter_y-1)):
#                 input_neuron_id = window_x+filter_coords[0] + (window_y+filter_coords[1])*x_input
#                 convo_neuron_id = window_x + window_y*(x_input-filter_x+1)
#                 if not np.isnan(self.DelayWeights.delay[input_neuron_id][convo_neuron_id]) and not np.isnan(self.DelayWeights.weight[input_neuron_id][convo_neuron_id]):
#                     self.DelayWeights.delay[input_neuron_id][convo_neuron_id] = self.filter_d[filter_coords[0]][filter_coords[1]]
#                     self.DelayWeights.weight[input_neuron_id][convo_neuron_id] = self.filter_w[filter_coords[0]][filter_coords[1]]

#     # Applies delta_d and delta_w to the whole filter 
#     def actualizeAllFilter(self, delta_d, delta_w):

#         for x in range(len(self.filter_d)):
#             for y in range(len(self.filter_d[x])):
#                 self.filter_d[x][y] = max(0.01, min(self.filter_d[x][y]+delta_d, self.max_delay))
#                 self.filter_w[x][y] = max(0.05, self.filter_w[x][y]+delta_w)

#         # Finally we actualize the weights and delays of all neurons that use the same filter
#         for window_x in range(0, x_input - (filter_x-1)):
#             for window_y in range(0, y_input - (filter_y-1)):
#                 for x in range(len(self.filter_d)):
#                     for y in range(len(self.filter_d[x])):
#                         input_neuron_id = window_x+x + (window_y+y)*x_input
#                         convo_neuron_id = window_x + window_y*(x_input-filter_x+1)
#                         if input_neuron_id < self.input.size and not np.isnan(self.DelayWeights.delay[input_neuron_id][convo_neuron_id]) and not np.isnan(self.DelayWeights.weight[input_neuron_id][convo_neuron_id]):
#                             self.DelayWeights.delay[input_neuron_id][convo_neuron_id] = self.filter_d[x][y]
#                             self.DelayWeights.weight[input_neuron_id][convo_neuron_id] = self.filter_w[x][y]

#     def get_filters(self):
#         return self.filter_d, self.filter_w

#     def check_activity_tags(self, neuron_to_check):
#         for conv in neuron_activity_tag:
#             if conv[neuron_to_check]:
#                 return False
#         return True


class LearningMechanisms(object):
    """
    Applies all learning mechanisms:
    - STDP on weights and Delays
    - Homeostasis
    - Threshold adaptation (not working)
    - checking for learning stop condition
    """
    
    def __init__(
        self, 
        sampling_interval, pattern_duration,
        input_spikes_recorder, output_spikes_recorder,
        projection, projection_delay_weight_recorder,
        A_plus, A_minus,
        B_plus, B_minus,
        tau_plus, tau_minus,
        teta_plus, teta_minus,
        filter_w, filter_d,
        stop_condition,
        growth_factor,
        Rtarget=0.0002, 
        lambda_w=0.0001, lambda_d=0.001, 
        thresh_adapt=False, label=0
    ):
        self.interval = sampling_interval
        self.pattern_duration = pattern_duration
        self.projection = projection
        self.input = projection.pre
        self.output = projection.post
        #self.input_last_spiking_times = [-1 for n in range(len(self.input))] # For aech neuron we keep its last time of spike
        #self.output_last_spiking_times = [-1 for n in range(len(self.output))]

        self.input_spikes = input_spikes_recorder
        self.output_spikes = output_spikes_recorder
        self.DelayWeights = projection_delay_weight_recorder

        """
        # We keep the last time of spike of each neuron
        self.input_last_spiking_times = self.input_spikes._spikes
        self.output_last_spiking_times = self.output_spikes._spikes
        """

        self.A_plus = A_plus
        self.A_minus = A_minus
        self.B_plus = B_plus
        self.B_minus = B_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.teta_plus = teta_plus
        self.teta_minus = teta_minus

        self.max_delay = False # If set to False, we will find the maximum delay on first call.
        self.filter_d = filter_d
        self.filter_w = filter_w
        self.c = stop_condition
        self.growth_factor = growth_factor
        self.label = label # Just the int associated with the layer theses mechanisms are applied to (0-3)
        self.thresh_adapt = thresh_adapt # Should be set to False (threshold adaptation not working)
        

        # For each neuron, we count their number of spikes to compute their activation rate.
        self.total_spike_count_per_neuron = [
            np.array([
                Rtarget for _ in range(10)
            ]) for _ in range(len(self.output))
        ] 

        # Number of times this has been called.
        self.call_count = 0 
        
        self.Rtarget = Rtarget
        self.lambda_w = lambda_w 
        self.lambda_d = lambda_d
        

    def __call__(self, t):
        
        global LEARNING
        if options.debug:
            print('> learning mechanisms')
            if LEARNING:
                print('>> learning phase done')
        
        if not LEARNING:
            self.learn(t)
    
    def learn(self,t):
        self.call_count += 1
        final_filters[self.label] = [self.filter_d, self.filter_w]

        input_spike_train = self.input_spikes._spikes
        output_spike_train = self.output_spikes._spikes

        """
        # We get the current delays and current weights
        delays = self.projection.get("delay", format="array")
        weights = self.projection.get("weight", format="array")
        => can be obtained using self.DelayWeights.delay and self.DelayWeights.weight
        """

        # The sum of all homeostasis delta_d and delta_t computed for each cell
        homeo_delays_total = 0
        homeo_weights_total = 0

        # Since we can't increase the delays past the maximum delay set at the beginning of the simulation,
        # we find the maximum delay during the first call
        if self.max_delay == False:
            self.max_delay = 0.01
            for x in self.DelayWeights.delay:
                for y in x:
                    if not np.isnan(y) and y > self.max_delay:
                        self.max_delay = y

        for post_neuron in range(self.output.size):

            # We only keep track of the activations of each neuron on a timeframe of 10 stimuli
            self.total_spike_count_per_neuron[post_neuron][int((t//self.interval)%len(self.total_spike_count_per_neuron[post_neuron]))] = 0

            # If the neuron spiked...
            if output_spike_train[post_neuron] != -1 and self.check_activity_tags(post_neuron):
                neuron_activity_tag[self.label][post_neuron] = True

                self.total_spike_count_per_neuron[post_neuron][int((t//self.interval)%len(self.total_spike_count_per_neuron[post_neuron]))] += 1

                # The neuron spiked during this stimulus and its threshold should be increased.
                # Since NEST won't allow neurons with a threshold > 0 to spike, we decrease v_rest instead.
                if self.thresh_adapt:
                    current_rest = self.output.__getitem__(post_neuron).get_parameters()['v_rest']
                    thresh = self.output.__getitem__(post_neuron).get_parameters()['v_thresh']
                    self.output.__getitem__(post_neuron).v_rest  = min(current_rest-(1.0-self.Rtarget), thresh-1)
                    self.output.__getitem__(post_neuron).v_reset = min(current_rest-(1.0-self.Rtarget), thresh-1)
                
                if options.verbose:
                    print("=== Neuron {} from layer {} spiked ! ===".format(post_neuron, self.label))            

                if not self.stop_condition(post_neuron):
                    # We actualize the last time of spike for this neuron
                    # self.output_last_spiking_times[post_neuron] = output_spike_train[post_neuron][-1]

                    # We now compute a new delay for each of its connections using STDP
                    for pre_neuron in range(len(self.DelayWeights.delay)):

                        # For each post synaptic neuron that has a connection with pre_neuron, 
                        # we also check that both neurons already spiked at least once.
                        if not np.isnan(self.DelayWeights.delay[pre_neuron][post_neuron]) and not np.isnan(self.DelayWeights.weight[pre_neuron][post_neuron]) and input_spike_train[pre_neuron] != -1:

                            # Some values here have a dimension in ms
                            delta_t = output_spike_train[post_neuron] - input_spike_train[pre_neuron] - self.DelayWeights.delay[pre_neuron][post_neuron]
                            delta_d = self.G(delta_t)
                            delta_w = self.F(delta_t)

                            """
                            convo_coords = [post_neuron%(x_input-filter_x+1), post_neuron//(x_input-filter_x+1)]
                            input_coords = [pre_neuron%x_input, pre_neuron//x_input]
                            filter_coords = [input_coords[0]-convo_coords[0], input_coords[1]-convo_coords[1]]
                            => never used
                            """

                            if options.verbose:
                                print("STDP from layer: {} with post_neuron: {} and pre_neuron: {} deltad: {}, deltat: {}".format(self.label, post_neuron, pre_neuron, delta_d*ms, delta_t*ms))
                            
                            self.actualize_filter(pre_neuron, post_neuron, delta_d, delta_w)

            # The neuron did not spike and its threshold should be lowered
            elif self.thresh_adapt:
                thresh = self.output.__getitem__(post_neuron).get_parameters()['v_thresh']
                current_rest = self.output.__getitem__(post_neuron).get_parameters()['v_rest']
                self.output.__getitem__(post_neuron).v_rest=min(current_rest+self.Rtarget, thresh-1)
                self.output.__getitem__(post_neuron).v_reset=min(current_rest+self.Rtarget, thresh-1)

            # Homeostasis regulation per neuron
            # R_observed = self.total_spike_count_per_neuron[post_neuron].sum()/self.call_count
            R_observed = self.total_spike_count_per_neuron[post_neuron].sum()/len(self.total_spike_count_per_neuron[post_neuron])
            K = (self.Rtarget - R_observed) / self.Rtarget

            if options.verbose:
                print("convo {} R: {}".format( self.label, R_observed))
            delta_d = - self.lambda_d * K
            delta_w =   self.lambda_w * K
            # Since weights and delays are shared, we can just add the homestatis deltas of all neurons add apply
            # the homeostasis only once after it has been computed for each neuron.
            homeo_delays_total  += delta_d  
            homeo_weights_total += delta_w 

        if options.verbose:
            print("****** CONVO {} homeo_delays_total: {}, homeo_weights_total: {}".format(self.label, homeo_delays_total, homeo_weights_total))
        
        self.actualize_All_Filter( 
            homeo_delays_total + self.growth_factor * self.pattern_duration, 
            homeo_weights_total)
        
        # At last we give the new delays and weights to our projections
        self.projection.set(delay = self.DelayWeights.delay)
        self.projection.set(weight = self.DelayWeights.weight)

        # We update the list that tells if this layer has finished learning the delays and weights
        full_stop_condition[self.label] = self.full_stop_check()
        # return t + self.interval

    # Computes the delay delta by applying the STDP
    def G(self, delta_t):
        if delta_t >= 0:
            delta_d = -self.B_minus*np.exp(-delta_t/self.teta_minus)
        else:
            delta_d = self.B_plus*np.exp(delta_t/self.teta_plus)
        return delta_d

    # Computes the weight delta by applying the STDP
    def F(self, delta_t):
        if delta_t >= 0:
            delta_w = self.A_plus*np.exp(-delta_t/self.tau_plus)
        else:
            delta_w = -self.A_minus*np.exp(delta_t/self.tau_minus)
        return delta_w

    # Given a post synaptic cell, returns if that cell has reached its stop condition for learning
    def stop_condition(self, post_neuron):
        min_ = 1e6
        for pre_neuron in range(len(self.DelayWeights.delay)):
            if not np.isnan(self.DelayWeights.delay[pre_neuron][post_neuron]) and self.DelayWeights.delay[pre_neuron][post_neuron] <= self.c:
                print("!!!!!!!!!!!!!!!!!!",pre_neuron, self.DelayWeights.delay[pre_neuron])
                return True
            if not np.isnan(self.DelayWeights.delay[pre_neuron][post_neuron]) and self.DelayWeights.delay[pre_neuron][post_neuron] < min_:
                min_ = self.DelayWeights.delay[pre_neuron][post_neuron]
        print("minimum delay",min_)
        return False

    # Checks if all cells have reached their stop condition
    def full_stop_check(self):
        for post_neuron in range(self.output.size):
            if not self.stop_condition(post_neuron):
                return False
        return True

    # Applies the current weights and delays of the filter to all the cells sharing those
    def actualize_filter(self, pre_neuron, post_neuron, delta_d, delta_w):

        # We now find the delay/weight to use by looking at the filter
        conv_coords   = [post_neuron%(x_input-filter_x+1), post_neuron//(x_input-filter_x+1)]
        input_coords  = [pre_neuron%x_input, pre_neuron//x_input]
        filter_coords = [input_coords[0] - conv_coords[0], input_coords[1] - conv_coords[1]]

        # And we actualize delay/weight of the filter after the STDP
        self.filter_d[filter_coords[0]][filter_coords[1]] = max(0.01, min(self.filter_d[filter_coords[0]][filter_coords[1]]+delta_d, self.max_delay))
        self.filter_w[filter_coords[0]][filter_coords[1]] = max(0.01, self.filter_w[filter_coords[0]][filter_coords[1]]+delta_w)

        coord_conv = self.get_convolution_window(post_neuron)
        diff = pre_neuron-coord_conv
        for post in range(len(self.output)):
            self.DelayWeights.delay[ self.get_convolution_window(post)+diff][post] = max(
                0.01, min(self.DelayWeights.delay[self.get_convolution_window(post)+diff][post]+delta_d, self.max_delay)
            )
            self.DelayWeights.weight[self.get_convolution_window(post)+diff][post] = max(
                0.01, self.DelayWeights.weight[self.get_convolution_window(post)+diff][post]+delta_w
            )


    # Applies delta_d and delta_w to the whole filter 
    # /!\ this method actually returns the new delays and weights
    def actualize_All_Filter(self, delta_d, delta_w):

        self.filter_d = np.where(
            (self.filter_d + delta_d < self.max_delay) & (self.filter_d > 0.01), 
            self.filter_d + delta_d, 
            self.filter_d
        )
        self.filter_w = np.where( 
            self.filter_w + delta_w > 0.01, 
            self.filter_w + delta_w, 
            self.filter_w
        )

        """
        delays = np.where(np.logical_not(np.isnan(delays)) & (delays + delta_d < self.max_delay) & (delays + delta_d > 0.01), delays+delta_d, np.maximum(0.01, np.minimum(self.max_delay, delays+delta_d)))
        weights = np.where(np.logical_not(np.isnan(weights)) & (weights + delta_w>0.01), weights+delta_w, np.maximum(0.01, self.max_delay, weights + delta_w))
        return delays.copy(), weights.copy()
        """
        
        self.DelayWeights.delay = np.where(
            np.logical_not(np.isnan(self.DelayWeights.delay)) & (self.DelayWeights.delay + delta_d < self.max_delay) & (self.DelayWeights.delay + delta_d > 0.01), 
            self.DelayWeights.delay + delta_d, 
            np.maximum(0.01, np.minimum(self.max_delay, self.DelayWeights.delay + delta_d))
        )
        self.DelayWeights.weight = np.where(
            np.logical_not(np.isnan(self.DelayWeights.weight)) & (self.DelayWeights.weight + delta_w > 0.01), 
            self.DelayWeights.weight + delta_w, 
            np.maximum(0.01, self.max_delay, self.DelayWeights.weight + delta_w)
        )

    # Given
    def get_convolution_window(self, post_neuron):
        return post_neuron//(x_input-filter_x+1)*x_input + post_neuron%(x_input-filter_x+1)

    def get_filters(self):
        return self.filter_d, self.filter_w

    def check_activity_tags(self, neuron_to_check):
        for conv in neuron_activity_tag:
            if conv[neuron_to_check]:
                return False
        return True


class visualiseFilters(object):
    def __init__(self, sampling_interval, results_path):
        self.interval = sampling_interval
        self.plot_delay_weight = []
        self.output_path = results_path
        os.makedirs(self.output_path, exist_ok=True)
        self.delay_matrix = []
        
        self.nb_call = 0
        # plot parameters
        self.log_str = ["Delays of convolution", "Weights of convolution"]
        self.color_map = plt.cm.autumn # https://matplotlib.org/stable/tutorials/colors/colormaps.html
        self.scale_x = filter_x + (NB_CONV_LAYERS - 2) * 4
        self.scale_y = filter_y + (NB_CONV_LAYERS - 2) * 4
        self.fontsize = 9 + 1.05*NB_CONV_LAYERS

    def __call__(self, t):
        if t > 0 and int(t) % pattern_interval == 0:
            self.display_filters(t)
    
    def compare(self, x_start, y_start, x_stop, y_stop):
        if x_start < x_stop :
            x_step = 1
        else:
            x_step = -1
        if y_start < y_stop :
            y_step = 1
        else:
            y_step = -1

        if x_start != x_stop and y_start != y_stop :
            for x, y in zip(range(x_start, x_stop, x_step), range(y_start, y_stop, y_step)):
                # if delay_matrix[y, x] > delay_matrix[y+1, x+1]:  # OK
                if self.delay_matrix[y, x] <= self.delay_matrix[y+y_step, x+x_step]:
                    return False
            return True

        elif x_start != x_stop:
            for x in range(x_start, x_stop, x_step):
                # if delay_matrix[y, x] > delay_matrix[y+1, x+1]:  # OK
                if self.delay_matrix[y_start, x] <= self.delay_matrix[y_start, x+x_step]:
                    return False
            return True

        elif y_start != y_stop:
            for y in range(y_start, y_stop, y_step):
                # if delay_matrix[y, x] > delay_matrix[y+1, x+1]:  # OK
                if self.delay_matrix[y, x_start] <= self.delay_matrix[y+y_step, x_start]:
                    return False
            return True
        
        else:
            return False

    def motion_recognition(self):
        """
        Return an int that indicates the direction in which the input delay matrix has specialized.
        For the moment, 9 possibles outputs:
        - 0 = (SOUTH-EAST ↘︎)
        - 1 = (SOUTH-WEST ↙︎)
        - 2 = (NORTH-WEST ↖︎)
        - 3 = (NORTH-EAST ↗︎)
        - 4 = (EAST →)
        - 5 = (SOUTH ↓)
        - 6 = (WEST ←)
        - 7 = (NORTH ↑)
        - -1 = (INDETERMINATE)
        """
        min_coord = 0
        half_coord = filter_x // 2
        max_coord = filter_x-1
        # coord_motions = {
        #     0: [[min_coord, min_coord],  [max_coord, max_coord]],
        #     1: [[max_coord, min_coord],  [min_coord, max_coord]],
        #     2: [[max_coord, max_coord],  [min_coord, min_coord]],
        #     3: [[min_coord, max_coord],  [max_coord, min_coord]],
        #     4: [[min_coord, half_coord], [max_coord, half_coord]],
        #     5: [[half_coord, min_coord], [half_coord, max_coord]],
        #     6: [[max_coord, half_coord], [min_coord, half_coord]],
        #     7: [[half_coord, max_coord], [half_coord, min_coord]]
        # }

        start_motions = {
            (min_coord, min_coord):  0,
            (max_coord, min_coord):  1,
            (max_coord, max_coord):  2,
            (min_coord, max_coord):  3,
            (min_coord, half_coord): 4,
            (half_coord, min_coord): 5,
            (max_coord, half_coord): 6,
            (half_coord, max_coord): 7
        }
        stop_motions = {  # key: coord start - value: corresponding coord stop
            min_coord: max_coord,
            max_coord: min_coord,
            half_coord:half_coord
        }

        coord = [min_coord, half_coord, max_coord]
        for x_start, y_start in it.product(coord, coord):
            if not (x_start ==2 and x_start == y_start):
                x_stop = stop_motions[x_start]
                y_stop = stop_motions[y_start]
                if self.compare(x_start, y_start, x_stop, y_stop):
                    return start_motions[(x_start, y_start)]
        return -1
    
    def display_filters(self,t):
        """
        Create and save a plot that contains for each convolution filter its delay matrix and associated weights of the current model state
        """
        global motion_per_conv
        file_name = os.path.join(self.output_path, 'delays_and_weights_'+str(self.nb_call)+'.png')
        fig, axs = plt.subplots(nrows=len(self.log_str), ncols=NB_CONV_LAYERS, sharex=True, figsize=(self.scale_x, self.scale_y))
        
        for n_log in range(len(self.log_str)):
            for n_layer in range(NB_CONV_LAYERS):
                self.delay_matrix = final_filters[n_layer][n_log]
                
                title = self.log_str[n_log] + ' ' + str(n_layer)
                if n_log == 0: # delay matrix part
                    id_motion = self.motion_recognition()
                    if id_motion != -1:
                        motion_per_conv[n_layer] = id_motion
                    """
                    else:
                        motion_per_conv.pop(n_layer, None)
                    """
                    title += '\n' + DIRECTIONS[id_motion] + '(' + str(id_motion) + ')'
                
                fig_matrix = axs[n_log][n_layer]
                fig_matrix.set_title(title, fontsize=self.fontsize)
                im_matrix = fig_matrix.imshow(self.delay_matrix, cmap = self.color_map)
                fig.colorbar(im_matrix, ax=fig_matrix, fraction=0.046, pad=0.04)
        
        fig.suptitle('Delays and Weights kernel at t:'+str(t), fontsize=self.fontsize)
        plt.tight_layout()
        fig.savefig(file_name, dpi=300)
        plt.close()

        self.plot_delay_weight.append(file_name)
        self.nb_call += 1
        if options.verbose:
            print("[", self.nb_call , "] : Images of delays and weights saved as", file_name)

    def print_final_filters(self):
        """
        Create a gif containing every images generated by print_filters
        """
        imgs = [imageio.imread(step_file) for step_file in self.plot_delay_weight]
        imageio.mimsave( os.path.join(self.output_path, 'delays_and_weights_evolution.gif'), imgs, duration=1) # 1s between each frame of the gif


class Metrics(object):
    def __init__(self, Conv_spikes, motion_per_conv):
        # The spikes produced by each convolution layer is stored in this dictionary
        # key: id convolution - value: spikes
        self.spikes_per_conv = {}
        self.Conv_spikes = Conv_spikes
        self.NB_CONV_LAYERS = len(Conv_spikes)
        self.motion_per_conv = motion_per_conv
        self.metrics = {}      # key: id convolution - value: [precision, recall, F1]
        self.gini = None

    def spiketrains2array(self):
        """
        Fill spikes_per_conv with every spike time produced in each convolution layers
        """
        for n_conv in range(self.NB_CONV_LAYERS): 
            array = np.array([])
            for spikes in self.Conv_spikes[n_conv]:
                array = np.concatenate(
                    (array, np.array(spikes)),
                    axis = 0
                )
            self.spikes_per_conv[n_conv] = np.sort(array)       

    def within_interval(self, interval, timestamps):
        """
        From an interval and a list of timestamps, 
        return True if at least one timestep is within the interval
        and Flase otherwise. 
        Timing must be after the beginning of the interval, and cannot exceed 40 (ms) after the end of the interval.

        E.g :
        matching_spikes((300, 800), [400, 300, 600, 900, 840, 839, 301, 275]) => True
        """
        [start, end] = interval
        for ts in timestamps:
            if ts > start and ts - 40 < end:
                return True
        return False

    def compute_metrics(self):
        """
        Compute the GINI, PRECISION, RECALL and F1-SCORE based on the spikes produced in each convolution layers. 
        We compare whether it was supposed to produce spike or not, in particular in observing the direction of the 
        input spike and the direction of the layer where the spike is produced, to get the number of TruePositive, FalsePositive, FalseNegative.

        - TruePositive  (TP): Spike produced on the right convolution layer
        - FalsePositive (FP): Spike produced on the convolution layer but direction does not match
        - FalseNegative (FN): No spike produced but the direction was good
        - ProbaConvo    (PC): Probability of direction d in convolution c, for each d in input events

        Then, 
        - PRECISION = TP / (TP + FP)
        - RECALL = TP / (TP + FN)
        - F1-SCORE = 2 * ((PRECISION * RECALL) / (PRECISION + RECALL))
        - Gini = 1 - Sum(ProbaConv for each convolution)²
        """

        self.spiketrains2array()

        conv_variables = {   # key: id convolution - value: [nb TP, nb FP, nb FN, nb PC]
            id_conv: [
                0,0,0,
                [0 for _ in range(NB_DIRECTIONS)]
            ]
            for id_conv in range(NB_CONV_LAYERS)
        }

        # get TP, FP, FN for each interval, for each convolution layer
        for id_conv, output_spikes in self.spikes_per_conv.items():
            if id_conv in self.motion_per_conv.keys():
                n_spikes = len(output_spikes)
                for interval, id_motion in input_data.items():
                    TP = FP = FN = 0
                    PC = [0 for _ in range(NB_DIRECTIONS)]
                    conv_motion = self.motion_per_conv[id_conv]
                    within = self.within_interval(interval, output_spikes)
                    if within: 
                        PC[id_motion] += 1
                        if id_motion == conv_motion:
                            TP += 1
                        else:
                            FP += 1
                    elif not within and id_motion == conv_motion:
                        FN += 1    # TODO: modify so that FN is increased by the number of spikes normally obtained (2,3 or more)
                    conv_variables[id_conv][:3] = [
                        sum(x) for x in zip(
                            conv_variables[id_conv][:3],
                            [TP, FP, FN]
                        )
                    ]
                    conv_variables[id_conv][3] = [
                        sum(x) for x in zip(
                            conv_variables[id_conv][3],
                            PC
                        )
                    ]
                conv_variables[id_conv][3] = [
                    (e / n_spikes)*(e / n_spikes) 
                    for e in conv_variables[id_conv][3]
                ]


        # computer metrics for each layer
        for id_conv, var in conv_variables.items():
            if id_conv in self.motion_per_conv.keys():
                [TP, FP, FN, PC] = var
                if TP == 0:   # to avoid exception "dividing by zero"
                    self.metrics[id_conv] = [0,0,0]
                else: 
                    precision = TP / (TP + FP)
                    recall = TP / (TP + FN)
                    F1 = 2 * ((precision * recall) / (precision + recall))
                    self.metrics[id_conv] = [precision, recall, F1]
                
                gini = 1 - np.sum(PC)
                self.metrics[id_conv].append(gini)

            else:
                self.metrics[id_conv] = ['NA']*4

    def display_metrics(self): 
        self.compute_metrics()
        for n_conv, met in self.metrics.items():
            print('Convolution', n_conv)
            print('- precision:', met[0])
            print('- recall:   ', met[1])
            print('- F1:       ', met[2])
            print('- GINI:     ', met[3])


class callbacks_(object):
    def __init__(self, sampling_interval):
        self.call_order = []
        self.interval = sampling_interval
        self.learning = False

    def add_object(self,obj):
        self.call_order.append(obj)

    def __call__(self,t):
        for obj in self.call_order:
            if t%obj.interval == 0 and t != 0:
                obj.__call__(t)
        
        # if LEARNING:
        #     return t + time_data           
        return t + self.interval


### Simulation parameters
"""
growth_factor = (0.001/pattern_interval)*pattern_duration # <- juste faire *duration dans STDP We increase each delay by this constant each step

# Stop Condition
c = 1.0

# STDP weight
A_plus = 0.05  
A_minus = 0.05
tau_plus= 1.0 
tau_minus= 1.0

# STDP delay (2.5 is good too)
B_plus = 5.0 
B_minus = 5.0
teta_plus = 1.0 
teta_minus = 1.0

STDP_sampling = pattern_interval
"""

growth_factor = 0.0001

# Stop Condition
c = 0.5

# STDP weight
A_plus = 0.01 
A_minus = 0.01
tau_plus= 1.0 
tau_minus= 1.0

# STDP delay (2.5 is good too)
# B_plus = 1.0
# B_minus = 1.0
B_plus = B_minus = 2.5
teta_plus = 1.0 
teta_minus = 1.0

STDP_sampling = pattern_interval

### Launch simulation

visu_time = visualiseTime(sampling_interval=500)
visu_filters = visualiseFilters(sampling_interval=500, results_path=results_path)

Input_spikes = LastSpikeRecorder(sampling_interval=STDP_sampling-1, pop=Input)
conv_spikes = []
for conv in convolutions:
    conv_spikes.append(
        LastSpikeRecorder(sampling_interval=STDP_sampling-1, pop=conv)
    )

input2conv_delay_weight = []
for conn in input2conv:
    input2conv_delay_weight.append(
        WeightDelayRecorder(sampling_interval=STDP_sampling, proj=conn)
    )

neuron_reset = NeuronReset(sampling_interval=STDP_sampling-5, pops=convolutions, t_pulse=5)
# neuron_reset = NeuronReset(sampling_interval=pattern_interval-15, pops=convolutions)
# input_clear = InputClear(sampling_interval=pattern_interval+1, pops_to_clear_data=Input)

learning_mechanisms = []
for idx in range(NB_CONV_LAYERS):
    learning_mechanisms.append(
        LearningMechanisms(
            sampling_interval=STDP_sampling, 
            pattern_duration=pattern_duration,
            input_spikes_recorder=Input_spikes, 
            output_spikes_recorder=conv_spikes[idx], 
            projection=input2conv[idx], 
            projection_delay_weight_recorder=input2conv_delay_weight[idx], 
            B_plus=B_plus, B_minus=B_minus, 
            tau_plus=tau_plus, tau_minus=tau_minus, 
            filter_d=delay_conv[idx], 
            A_plus=A_plus, A_minus=A_minus, 
            teta_plus=teta_plus, teta_minus=teta_minus, 
            filter_w=weight_conv[idx], 
            stop_condition=c, 
            growth_factor=growth_factor, 
            label=idx,
            ##########
            Rtarget=0.003,
            lambda_d=0.0006,
            lambda_w=0.00003,
        )
    ) 

callbacks = callbacks_(sampling_interval=1)
callback_list = [visu_time, neuron_reset, Input_spikes, *conv_spikes, *input2conv_delay_weight , *learning_mechanisms, visu_filters]

for obj in callback_list:
    callbacks.add_object(obj)
sim.run(time_data, callbacks=[callbacks])
# sim.run(time_data, callbacks=[visu, wd_rec, Input_spikes, Conv1_spikes, Conv2_spikes, input2conv1_delay_weight, input2conv2_delay_weight, neuron_reset, Learn1, Learn2])

run_time = dt.now() - start
print("complete simulation run time:", run_time)


if options.save: 
    options.metrics = True

if options.plot_figure or options.metrics:
    Conv_spikes = [conv.get_spikes() for conv in conv_spikes]

### Plot figure

if options.plot_figure :
    extension = '_'+str(NB_DIRECTIONS)+'directions_'+str(NB_CONV_LAYERS)+'convolutions'
    title = 'Delay learning - '+str(NB_DIRECTIONS)+' directions '+str(NB_CONV_LAYERS)+' convolutions'

    conv_data = [conv.get_data() for conv in convolutions]
    Input_spikes = Input_spikes.get_spikes()
    
    # figure_filename = normalized_filename("Results", "delay_learning"+extension, "png", options.simulator)
    figure_filename = os.path.join(results_path, "delay_learning"+extension, "png")

    figure_params = []
    # Add reaction neurons spike times
    for i in range(NB_CONV_LAYERS):
        direction_id_of_conv = -1 if i not in motion_per_conv else motion_per_conv[i]
        direction_str = DIRECTIONS[direction_id_of_conv]
        figure_params.append(Panel(Conv_spikes[i], xlabel="Convolution "+str(i)+" spikes - "+direction_str+"("+str(direction_id_of_conv)+")", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, convolutions[i].size)))

    if NB_DIRECTIONS == 2:
        Figure(
            # raster plot of the event inputs spike times
            Panel(Input_spikes, xlabel="Input spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, Input.size)),
            # raster plot of the Reaction neurons spike times
            Panel(Conv_spikes[0], xlabel="Convolution 1 spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, convolutions[0].size)),
            # raster plot of the Output1 neurons spike times
            Panel(Conv_spikes[1], xlabel="Convolution 2 spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, convolutions[1].size)),
            title=title,
            annotations="Simulated with "+ options.simulator.upper()
        ).save(figure_filename)

    else:
        Figure(
            # raster plot of the event inputs spike times
            Panel(Input_spikes, xlabel="Input spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, Input.size)),
            *figure_params,
            title=title,
            annotations="Simulated with "+ options.simulator.upper()
        ).save(figure_filename)

    visu_filters.print_final_filters() 

    print("Figures correctly saved as", figure_filename)
    # plt.show()


### Display metrics
if options.metrics:
    # if len(motion_per_conv) != NB_DIRECTIONS: #TODO: replace by NB_CONV_LAYERS ?
    #     print('Error: cannot compute metrics. The convolution layers did not converge sufficiently towards a direction', flush=True)
    # else: 
    print('Computing metrics...', flush=True)
    # Get time of all spikes produced in each convolution layers
    metrics = Metrics(Conv_spikes, motion_per_conv)
    metrics.display_metrics()
    

### Save in csv
if options.save: 
    file_results = os.path.join(OUTPUT_PATH_GENERIC, 'results.csv')
    if not os.path.exists(file_results):
        header = 'simulation;nb convolutions;noise;length input events (in microseconds);run time;learning time;id convolution;id motion;direction;precision;recall;F1;gini;\n'
        print('header')
    else: 
        header = ''
    file = open(file_results,'a')
    file.write(header)

    # header = 'simulation;nb convolutions;noise;length input events (in microseconds);run time;learning time;id convolution;id direction;direction;precision;recall;F1;gini;\n'
    default_content = ';'.join([time_now, str(NB_CONV_LAYERS), str(options.noise),str(options.t), str(run_time), str(learning_time)])+';'
    
    # try:
    res_metrics = metrics.metrics
    content = ''
    for n_conv in range(NB_CONV_LAYERS):
        direction_id_of_conv = -1 if n_conv not in motion_per_conv else motion_per_conv[n_conv]
        print(direction_id_of_conv, DIRECTIONS)
        direction_str = DIRECTIONS[direction_id_of_conv]
        content += default_content + ';'.join([str(n_conv), str(direction_id_of_conv), direction_str]+[str(e) for e in res_metrics[n_conv]]) + ';\n'
    # except:
    #     content = default_content + ';'.join(['NA' for _ in range(7)]) +';\n'

    file.write(content)
    file.close()
    print('Results saved in', file_results)