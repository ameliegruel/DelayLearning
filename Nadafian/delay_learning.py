#!/bin/python
import itertools as it
from pyNN.utility import get_simulator, init_logging, normalized_filename
from pyNN.utility.plotting import Figure, Panel
from quantities import ms
from random import randint
import matplotlib.pyplot as plt
from datetime import datetime as dt
import neo
import numpy as np
from events2spikes import ev2spikes

start = dt.now()

sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file", {"action": "store_true"}),
                             ("--two", "Use only 2 layers instead of 4", {"action": "store_true"}),
                             ("--debug", "Print debugging information"))

if options.debug:
    init_logging(None, debug=True)

if sim == "nest":
    from pyNN.nest import *

sim.setup(timestep=0.01)

### Generate input data

time_data = 1e4
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

input_events = np.zeros((0,4))
for t in range(int(time_data/pattern_interval)):
    direction = randint(0,1)
    if direction==0:
        start_x = randint(x_margin, x_input-pattern_duration-x_margin) # We leave a margin of 4 neurons on the edges of the input layer so that the whole movement can be seen by the convolution window
        start_y = randint(y_margin, y_input-pattern_duration-y_margin)
        input_events = np.concatenate((
            input_events,
            [[start_x+d, start_y+d, 1, d+t*pattern_interval] for d in range(pattern_duration)]
        ), axis=0)
    
    elif direction==1:
        start_x = randint(x_input-x_margin-1, x_input-pattern_duration) # We leave a margin of 4 neurons on the edges of the input layer so that the whole movement can be seen by the convolution window
        start_y = randint(y_margin, y_input-pattern_duration-y_margin)
        input_events = np.concatenate((
            input_events,
            [[start_x-d, start_y+d, 1, d+t*pattern_interval] for d in range(pattern_duration)]
        ), axis=0)
        
input_spiketrain, _, _ = ev2spikes(input_events, width=x_input, height=y_input)


### Build Network 

# Populations

Input = sim.Population(
    x_input*y_input,  
    sim.SpikeSourceArray(spike_times=input_spiketrain), 
    label="Input"
)
Input.record("spikes")

Convolutions_parameters = {
    'tau_m': 20.0,       # membrane time constant (in ms)   
    'tau_refrac': 30.0,  # duration of refractory period (in ms) 0.1 de base
    'v_reset': -70.0,    # reset potential after a spike (in mV) 
    'v_rest': -70.0,     # resting membrane potential (in mV)
    'v_thresh': -5.0,    # spike threshold (in mV) -5 de base
}

# The size of a convolution layer with a filter of size x*y is input_x-x+1 * input_y-y+1 
Conv1 = sim.Population(
    x_output*y_output, 
    sim.IF_cond_exp(**Convolutions_parameters),
)
Conv1.record(('spikes','v'))

Conv2 = sim.Population(
    x_output*y_output,
    sim.IF_cond_exp(**Convolutions_parameters), 
)
Conv2.record(('spikes','v'))


# List connector

weight_N = 0.35 
delays_N = 15.0 
weight_teta = 0.005 
delays_teta = 0.05 

weight_conv = np.random.normal(weight_N, weight_teta, size=(2, filter_x, filter_y))
delay_conv =  np.random.normal(delays_N, delays_teta, size=(2, filter_x, filter_y))

input2output1_conn = []
input2output2_conn = []
c = 0

for in2out_conn in [input2output1_conn, input2output2_conn]:

    for X,Y in it.product(range(x_output), range(y_output)):

        idx = np.ravel_multi_index( (X,Y) , (x_output, y_output) )

        conn = []
        for x, y in it.product(range(filter_x), range(filter_y)):
            w = weight_conv[c, x, y]
            d = delay_conv[ c, x, y]
            A = np.ravel_multi_index( (X+x,Y+y) , (x_input, y_input) )
            conn.append( ( A, idx, w, d ) )

        in2out_conn += conn
    
    c += 1

# Projections

input2conv1 = sim.Projection(
    Input, Conv1,
    connector = sim.FromListConnector(input2output1_conn),
    synapse_type = sim.StaticSynapse(),
    receptor_type = 'excitatory',
    label = 'Input to Conv1'
)

input2conv2 = sim.Projection(
    Input, Conv2,
    connector = sim.FromListConnector(input2output2_conn),
    synapse_type = sim.StaticSynapse(),
    receptor_type = 'excitatory',
    label = 'Input to Conv2'
)

conv12conv2 = sim.Projection(
    Conv1, Conv2,
    connector = sim.OneToOneConnector(),
    synapse_type = sim.StaticSynapse(
        weight=50,
        delay=0.01
    ),
    receptor_type = "inhibitory",
    label = "Lateral inhibition - conv1 to conv2"
)

conv22conv1 = sim.Projection(
    Conv2, Conv1,
    connector = sim.OneToOneConnector(),
    synapse_type = sim.StaticSynapse(
        weight=50,
        delay=0.01
    ),
    receptor_type = "inhibitory",
    label = "Lateral inhibition - conv2 to conv1"
)

# We will use this list to know which convolution layer has reached its stop condition
full_stop_condition= [False, False]
# Each filter of each convolution layer will be put in this list and actualized at each stimulus
final_filters = [[], []]
# Sometimes, even with lateral inhibition, two neurons on the same location in different convolution
# layers will both spike (due to the minimum delay on those connections). So we keep track of
# which neurons in each layer has already spiked for this stimulus. (Everything is put back to False at the end of the stimulus)
neuron_activity_tag = [ [False for cell in range((x_input-filter_x+1)*(y_input-filter_y+1))]for conv in range(len(full_stop_condition)) ]


### Run simulation

# Callback classes

class LastSpikeRecorder(object):

    def __init__(self, sampling_interval, pop):
        self.interval = sampling_interval
        self.population = pop

        if type(self.population) != list:
            self._spikes = np.ones(self.population.size) * (-1)
        else:
            self._spikes = np.ones(len(self.population)) * (-1)

    def __call__(self, t):
        if t > 0:
            if type(self.population) != list:
                self._spikes = map(
                    lambda x: x[-1].item() if len(x) > 0 else -1, 
                    self.population.get_data("spikes", clear=True).segments[0].spiketrains
                )
                self._spikes = np.fromiter(self._spikes, dtype=float)

            else:
                self._spikes = []
                for subr in self.population:
                    sp = subr.get_data("spikes", clear=True).segments[0].spiketrains
                    spikes_subr = map(
                        lambda x: x[-1].item() if len(x) > 0 else -1, 
                        sp
                    )
                    self._spikes.append(max(spikes_subr))

        return t+self.interval

class WeightDelayRecorder(object):

    def __init__(self, sampling_interval, proj):
        self.interval = sampling_interval
        self.projection = proj

        self.weight = None
        self._weights = []
        self.delay = None
        self._delays = []

    def __call__(self, t):
        attribute_names = self.projection.synapse_type.get_native_names('weight','delay')
        self.weight, self.delay = self.projection._get_attributes_as_arrays(attribute_names, multiple_synapses='sum')
        
        self._weights.append(self.weight)
        self._delays.append(self.delay)

        return t+self.interval

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
    def __init__(self, sampling_interval, projection):
        self.interval = sampling_interval
        self.projection = projection

    def __call__(self, t):
        print("step : {}".format(t))

        if full_stop_condition[0] and full_stop_condition[1]:
            print("!!!! FINISHED LEARNING !!!!") 
            sim.end()
            self.print_final_filters()
            exit()
        if t > 1 and int(t) % pattern_interval==0:
            self.print_final_filters()

        return t + self.interval


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

        print("\n\n")
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

    def __init__(self, sampling_interval, pops):
        self.interval = sampling_interval
        self.populations = pops 

    def __call__(self, t):
        for conv in neuron_activity_tag:
            for cell in range(len(conv)):
                conv[cell] = False

        if t > 0:
            print("!!! RESET !!!")
            if type(self.populations)==list:
                for pop in self.populations:
                    pulse = sim.DCSource(amplitude=-10.0, start=t, stop=t+10)
                    pulse.inject_into(pop)
            else:
                pulse = sim.DCSource(amplitude=-10.0, start=t, stop=t+10)
                pulse.inject_into(self.populations)

            self.interval = pattern_interval
        return t + self.interval


class InputClear(object):
    """
    When called, simply gets the data from the input with the 'clear' parameter set to True.
    By periodically clearing the data from the populations the simulation goes a lot faster.
    """

    def __init__(self, sampling_interval, pops_to_clear_data):
        self.interval = sampling_interval
        self.pop_clear = pops_to_clear_data

    def __call__(self, t):
        if t > 0:
            print("!!! INPUT CLEAR !!!")
            try:
                input_spike_train = self.pop_clear.get_data("spikes", clear=True).segments[0].spiketrains 
            except:
                pass
            self.interval = pattern_interval
        return t + self.interval


class LearningMechanisms(object):
    def __init__(
        self, 
        sampling_interval, 
        input_spikes_recorder, output_spikes_recorder,
        projection, projection_delay_weight_recorder,
        B_plus, B_minus, 
        tau_plus, tau_minus, 
        A_plus, A_minus, 
        teta_plus, teta_minus, 
        filter_d, filter_w, 
        stop_condition, 
        growth_factor, 
        Rtarget=0.005, 
        lamdad=0.002, lamdaw=0.00005, 
        thresh_adapt=True, 
        label=0
    ):
        self.interval = sampling_interval
        self.projection = projection
        self.input = projection.pre
        self.output = projection.post

        self.input_spikes = input_spikes_recorder 
        self.output_spikes = output_spikes_recorder
        self.DelayWeights = projection_delay_weight_recorder
        
        # We keep the last time of spike of each neuron
        self.input_last_spiking_times = self.input_spikes._spikes
        self.output_last_spiking_times = self.output_spikes._spikes
        
        self.B_plus = B_plus
        self.B_minus = B_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.max_delay = False # If set to False, we will find the maximum delay on first call.
        self.filter_d = filter_d
        self.filter_w = filter_w
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.teta_plus = teta_plus
        self.teta_minus = teta_minus
        self.c = stop_condition
        self.growth_factor = growth_factor
        self.label = label
        self.thresh_adapt=thresh_adapt
        
        # For each neuron, we count their number of spikes to compute their activation rate.
        self.total_spike_count_per_neuron = [0 for _ in range(len(self.output))] 
        
        # Number of times this has been called.
        self.call_count = 0 
        
        self.Rtarget = Rtarget
        self.lamdaw = lamdaw 
        self.lamdad = lamdad

    def __call__(self, t):

        if t == 0 :
            print("No data")
            return t + pattern_interval

        self.call_count += 1
        final_filters[self.label] = [self.filter_d, self.filter_w]

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

        for pre_neuron in range(self.input.size):
            if self.input_spikes._spikes[pre_neuron] != -1 and self.input_spikes._spikes[pre_neuron] > self.input_last_spiking_times[pre_neuron]:
                # We actualize the last time of spike for this neuron
                self.input_last_spiking_times[pre_neuron] = self.input_spikes._spikes[pre_neuron]
                print("PRE SPIKE {} : {}".format(pre_neuron, self.input_spikes._spikes[pre_neuron]))


        for post_neuron in range(self.output.size):

            if self.output_spikes._spikes[post_neuron] != -1 and self.check_activity_tags(post_neuron):
                neuron_activity_tag[self.label][post_neuron] = True
                print("***** STIMULUS {} *****".format(t//pattern_interval))

                self.total_spike_count_per_neuron[post_neuron] += 1

                # The neuron spiked during this stimulus and its threshold should be increased.
                # Since Nest won't allow neurons with a threshold > 0 to spike, we decrease v_rest instead.
                current_rest = self.output.__getitem__(post_neuron).get_parameters()['v_rest']
                if self.thresh_adapt:
                    self.output.__getitem__(post_neuron).v_rest=current_rest-(1.0-self.Rtarget)
                    self.output.__getitem__(post_neuron).v_reset=current_rest-(1.0-self.Rtarget)
                print("=== Neuron {} from layer {} spiked ! Whith rest = {} ===".format(post_neuron, self.label, current_rest))
                print("Total pikes of neuron {} from layer {} : {}".format(post_neuron, self.label, self.total_spike_count_per_neuron[post_neuron]))

                if self.output_spikes._spikes[post_neuron] > self.output_last_spiking_times[post_neuron] and not self.stop_condition(post_neuron):
                    # We actualize the last time of spike for this neuron
                    self.output_last_spiking_times[post_neuron] = self.output_spikes._spikes[post_neuron]

                    # We now compute a new delay for each of its connections using STDP
                    for pre_neuron in range(len(self.DelayWeights.delay)):
                        
                        # For each post synaptic neuron that has a connection with pre_neuron, we also check that both neurons
                        # already spiked at least once.
                        if not np.isnan(self.DelayWeights.delay[pre_neuron][post_neuron]) and not np.isnan(self.DelayWeights.weight[pre_neuron][post_neuron]) and self.input_last_spiking_times[pre_neuron] != -1 and self.output_last_spiking_times[post_neuron] != -1:

                            # Some values here have a dimension in ms
                            delta_t = self.output_last_spiking_times[post_neuron] - self.input_last_spiking_times[pre_neuron] - self.DelayWeights.delay[pre_neuron][post_neuron]
                            delta_d = self.G(delta_t)
                            delta_w = self.F(delta_t)

                            print("STDP from layer: {} with post_neuron: {} and pre_neuron: {} deltad: {}, deltat: {}".format(self.label, post_neuron, pre_neuron, delta_d*ms, delta_t*ms))
                            print("TIME PRE {} : {} TIME POST 0: {} DELAY: {}".format(pre_neuron, self.input_last_spiking_times[pre_neuron], self.output_last_spiking_times[post_neuron], self.DelayWeights.delay[pre_neuron][post_neuron]))
                            self.actualize_filter(pre_neuron, post_neuron, delta_d, delta_w)
            else:
                # The neuron did not spike and its threshold should be lowered

                if self.thresh_adapt:
                    current_rest = self.output.__getitem__(post_neuron).get_parameters()['v_rest']
                    self.output.__getitem__(post_neuron).v_rest=current_rest+self.Rtarget
                    self.output.__getitem__(post_neuron).v_reset=current_rest+self.Rtarget

            # Homeostasis regulation per neuron
            Robserved = self.total_spike_count_per_neuron[post_neuron]/self.call_count
            K = (self.Rtarget - Robserved)/self.Rtarget
            delta_d = -self.lamdad*K
            delta_w = self.lamdaw*K
            homeo_delays_total += delta_d  
            homeo_weights_total += delta_w 
            print("Rate of neuron {} from layer {}: {}".format(post_neuron, self.label, Robserved))


        print("****** CONVO {} homeo_delays_total: {}, homeo_weights_total: {}".format(self.label, homeo_delays_total, homeo_weights_total))
        self.actualizeAllFilter( homeo_delays_total+self.growth_factor*self.interval, homeo_weights_total)

        # At last we give the new delays and weights to our projections
        
        self.DelayWeights.update_delays(self.DelayWeights.delay)
        self.DelayWeights.update_weights(self.DelayWeights.weight)
        
        ### HERE MODIFY 
        self.projection.set(delay = self.DelayWeights.delay)
        self.projection.set(weight = self.DelayWeights.weight)

        # We update the list that tells if this layer has finished learning the delays and weights
        full_stop_condition[self.label] = self.full_stop_check()
        return t + pattern_interval

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
        for pre_neuron in range(len(self.DelayWeights.delay)):
            if not np.isnan(self.DelayWeights.delay[pre_neuron][post_neuron]) and self.DelayWeights.delay[pre_neuron][post_neuron] <= self.c:
                return True
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
        convo_coords = [post_neuron%(x_input-filter_x+1), post_neuron//(x_input-filter_x+1)]
        input_coords = [pre_neuron%x_input, pre_neuron//x_input]
        filter_coords = [input_coords[0]-convo_coords[0], input_coords[1]-convo_coords[1]]

        # And we actualize delay/weight of the filter after the STDP
        print(pre_neuron, post_neuron)
        print(np.unravel_index(pre_neuron, (13,13)))
        print(np.unravel_index(post_neuron, (9,9)))
        print(input_coords, convo_coords, filter_coords)
        print(self.filter_d.shape, self.filter_w.shape)
        self.filter_d[filter_coords[0]][filter_coords[1]] = max(0.01, min(self.filter_d[filter_coords[0]][filter_coords[1]]+delta_d, self.max_delay))
        self.filter_w[filter_coords[0]][filter_coords[1]] = max(0.05, self.filter_w[filter_coords[0]][filter_coords[1]]+delta_w)

        # Finally we actualize the weights and delays of all neurons that use the same filter
        for window_x in range(0, x_input - (filter_x-1)):
            for window_y in range(0, y_input - (filter_y-1)):
                input_neuron_id = window_x+filter_coords[0] + (window_y+filter_coords[1])*x_input
                convo_neuron_id = window_x + window_y*(x_input-filter_x+1)
                if not np.isnan(self.DelayWeights.delay[input_neuron_id][convo_neuron_id]) and not np.isnan(self.DelayWeights.weight[input_neuron_id][convo_neuron_id]):
                    self.DelayWeights.delay[input_neuron_id][convo_neuron_id] = self.filter_d[filter_coords[0]][filter_coords[1]]
                    self.DelayWeights.weight[input_neuron_id][convo_neuron_id] = self.filter_w[filter_coords[0]][filter_coords[1]]

    # Applies delta_d and delta_w to the whole filter 
    def actualizeAllFilter(self, delta_d, delta_w):

        for x in range(len(self.filter_d)):
            for y in range(len(self.filter_d[x])):
                self.filter_d[x][y] = max(0.01, min(self.filter_d[x][y]+delta_d, self.max_delay))
                self.filter_w[x][y] = max(0.05, self.filter_w[x][y]+delta_w)

        # Finally we actualize the weights and delays of all neurons that use the same filter
        for window_x in range(0, x_input - (filter_x-1)):
            for window_y in range(0, y_input - (filter_y-1)):
                for x in range(len(self.filter_d)):
                    for y in range(len(self.filter_d[x])):
                        input_neuron_id = window_x+x + (window_y+y)*x_input
                        convo_neuron_id = window_x + window_y*(x_input-filter_x+1)
                        if input_neuron_id < self.input.size and not np.isnan(self.DelayWeights.delay[input_neuron_id][convo_neuron_id]) and not np.isnan(self.DelayWeights.weight[input_neuron_id][convo_neuron_id]):
                            self.DelayWeights.delay[input_neuron_id][convo_neuron_id] = self.filter_d[x][y]
                            self.DelayWeights.weight[input_neuron_id][convo_neuron_id] = self.filter_w[x][y]

    def get_filters(self):
        return self.filter_d, self.filter_w

    def check_activity_tags(self, neuron_to_check):
        for conv in neuron_activity_tag:
            if conv[neuron_to_check]:
                return False
        return True


### Simulation parameters

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

### Launch simulation

visu = visualiseTime(sampling_interval=500, projection=input2conv1)
wd_rec = WeightDelayRecorder(sampling_interval=1.0, proj=input2conv1)

Input_spikes = LastSpikeRecorder(sampling_interval=STDP_sampling, pop=Input)
Conv1_spikes = LastSpikeRecorder(sampling_interval=STDP_sampling, pop=Conv1)
Conv2_spikes = LastSpikeRecorder(sampling_interval=STDP_sampling, pop=Conv2)

input2conv1_delay_weight = WeightDelayRecorder(sampling_interval=STDP_sampling, proj=input2conv1)
input2conv2_delay_weight = WeightDelayRecorder(sampling_interval=STDP_sampling, proj=input2conv1)

neuron_reset = NeuronReset(sampling_interval=pattern_interval-15, pops=[Conv1, Conv2])
# input_clear = InputClear(sampling_interval=pattern_interval+1, pops_to_clear_data=Input)

Learn1 = LearningMechanisms(sampling_interval=STDP_sampling, input_spikes_recorder=Input_spikes, output_spikes_recorder=Conv1_spikes, projection=input2conv1, projection_delay_weight_recorder=input2conv1_delay_weight, B_plus=B_plus, B_minus=B_minus, tau_plus=tau_plus, tau_minus=tau_minus, filter_d=delay_conv[0], A_plus=A_plus, A_minus=A_minus, teta_plus=teta_plus, teta_minus=teta_minus, filter_w=weight_conv[0] , stop_condition=c, growth_factor=growth_factor, label=0)
Learn2 = LearningMechanisms(sampling_interval=STDP_sampling, input_spikes_recorder=Input_spikes, output_spikes_recorder=Conv2_spikes, projection=input2conv2, projection_delay_weight_recorder=input2conv2_delay_weight, B_plus=B_plus, B_minus=B_minus, tau_plus=tau_plus, tau_minus=tau_minus, filter_d=delay_conv[1], A_plus=A_plus, A_minus=A_minus, teta_plus=teta_plus, teta_minus=teta_minus, filter_w=weight_conv[1], stop_condition=c, growth_factor=growth_factor, label=1)

sim.run(time_data, callbacks=[visu, wd_rec, Input_spikes, Conv1_spikes, Conv2_spikes, input2conv1_delay_weight, input2conv2_delay_weight, neuron_reset, Learn1, Learn2])

print("complete simulation run time:", dt.now() - start)

### Plot figure

if options.plot_figure :

    if options.two :
        extension = '_2directions'
        title = "Delay learning - 2 directions"
    else : 
        extension = '_4directions'
        title = "Delay learning - 4 directions"
    
    Input_data = Input.get_data().segments[0]
    Conv1_data = Conv1.get_data().segments[0]
    Conv2_data = Conv2.get_data().segments[0]
    
    figure_filename = normalized_filename("Results", "delay_learning"+extension, "png", options.simulator)

    Figure(
        # raster plot of the event inputs spike times
        Panel(Input_data.spiketrains, xlabel="Input spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, Input.size)),
        # raster plot of the Reaction neurons spike times
        Panel(Conv1_data.spiketrains, xlabel="Conv1 spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, Conv1.size)),
        # raster plot of the Output1 neurons spike times
        Panel(Conv2_data.spiketrains, xlabel="Conv2 spikes", yticks=True, markersize=0.2, xlim=(0, time_data), ylim=(0, Conv2.size)),

        # membrane potential of the Conv1 neurons
        Panel(Conv1_data.filter(name='v')[0], xlabel="Membrane potential (mV)\nConv2 layer", yticks=True, xlim=(0, time_data), linewidth=0.2, legend=False),
        # membrane potential of the Conv2 neurons
        Panel(Conv2_data.filter(name='v')[0], xlabel="Membrane potential (mV)\nConv2 layer", yticks=True, xlim=(0, time_data), linewidth=0.2, legend=False),

        title=title,
        annotations="Simulated with "+ options.simulator.upper()
    ).save(figure_filename)
    print("Figures correctly saved as", figure_filename)
    plt.show()
