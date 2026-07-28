"""
core -- The spiking substrate: neurons, synapses, regions, and the Brain
they compose into, plus save/load and the optional torch backend.

Modules
-------
backend      Optional torch/CUDA/MPS population -- the same biology, faster.
             Never used to train anything: no autograd, no optimiser, no loss
biophysical  Continuous-time HH/AdEx circuit
brain        The simulation environment: regions + projections + world
dendrite     Dendritic trees, ~7000 syn/neuron, supralinear NMDA
neuron       Izhikevich neurons, SoA population, 7 cell types
persistence  gzip+pickle save/load. WARNING: unauthenticated pickle, see
             AUDIT.md A6
region       Functional area wrapper (sensory/assoc/motor/memory)
synapse      Projections, STDP, delays, tagging
"""
