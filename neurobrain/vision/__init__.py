"""
vision -- Early and mid visual cortex: the rate-coded ventral stream, the wide
retinotopic spiking V1, and the binding layers built on top of it.

Modules
-------
composite         Overlapping shapes that *require* binding
spikinghierarchy  Spiking V1→V2→V3 + active causal discovery
spikingvision     Real spiking retina→features→category, honest cost
v2binding         V2 units tuned to pairs of V1 features
ventral           Rate-coded ventral stream V1->V2->V4->IT, Gabors, stimuli
                  (the OLDER rate model -- not widev1)
widev1            Wide retinotopic spiking V1, 50 ms window, motion
"""
