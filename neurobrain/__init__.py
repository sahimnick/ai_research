"""
NeuroBrain
==========

A small, readable, biologically-inspired **spiking brain** you can teach and
watch think.

    from neurobrain import build_default_brain, teach_starter_curriculum
    from neurobrain import NetworkVisualizer

    brain = build_default_brain()
    teacher, encoders, report = teach_starter_curriculum(brain)

    frames = brain.present("text", encoders["text"].encode("light"))
    NetworkVisualizer(brain).animate(frames, save_path="run.gif")

Every public name lives on the package root, so the import above is the only
form you need. The subpackages below describe *where* things live, not how to
reach them.

Package layout
--------------
    neurobrain/
        workspace.py    the spine -- one shared code, alone at the root
        core/           the spiking substrate
        sensing/        world -> signal
        vision/         early + mid visual cortex
        audition/       early + mid auditory cortex
        world/          what is out there, and what happens next
        memory/         storing, replaying, reconstructing
        cognition/      the higher faculties
        learning/       teaching, curricula, self-organisation
        minds/          whole assembled systems
        viz/            watch it think
        tools/          drive it, diagnose it, measure it

Each subpackage's ``__init__`` lists its own modules with a one-line role.

workspace
~~~~~~~~~
workspace    ONE shared cortical code every faculty projects into and reads
             from; names are a late read-out of a code, never its substance.

core/ -- the spiking substrate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
neuron       Izhikevich neuron model + neuron *types* (the multiple models).
synapse      Connections + STDP learning (Hebbian, no backprop).
region       Functional brain areas (sensory / association / motor / memory).
brain        The brain environment: regions + world model + recorder.
dendrite     Dendritic tree: many branches x synapses -> biological fan-in
             (~7000 synapses/neuron) with supralinear NMDA integration.
biophysical  Continuous-time HH/AdEx circuit + synaptic growth + shortcuts.
backend      Optional torch/CUDA/MPS population -- the same biology, faster;
             never used to train anything.
persistence  gzip+pickle save/load of a whole brain.

sensing/ -- world -> signal
~~~~~~~~~~~~~~~~~~~~~~~~~~~
realworld    Self-grown categories on REAL data (MNIST) + 'don't know' (OOD).
sensors      Encoders: text / image / sound / vector -> neural currents.
streams      Continuous, unsegmented REAL streams: a saccadic eye with a fovea
             on a large cluttered scene, an ear on a soundscape with no marked
             boundaries, and the two bound in the one shared code.

vision/ -- early + mid visual cortex
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ventral      Spiking ventral visual stream: V1->V2->V4->IT + top-down attention.
             (The older RATE-coded stream -- not to be confused with widev1.)
widev1       Width instead of depth: a wide retinotopic spiking V1 (10x10
             fields, up to 4096 cells), a 50 ms windowed firing rate, and
             motion coding measured against a control that cannot cheat.
spikingvision Real MNIST perceived by ACTUAL Izhikevich spiking neurons (rate
             coding + spike-count categories), with the honest cost measured.
spikinghierarchy  Spiking V1->V2->V3 ventral stream + active causal discovery
             (acting where the model is most uncertain, not at random).
v2binding    V2: units tuned to PAIRS of V1 features in a spatial and angular
             relation -- edge + edge + orientation = shape fragment.
composite    Overlapping shapes whose classes SHARE their parts -- a task that
             really needs binding -- and a topographic local V2 that does it.

audition/ -- early + mid auditory cortex
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
audio        Spiking auditory hierarchy: cochlea -> A1 -> belt.
auditorycortex  A contrastive-predictive A1: frequency-by-time receptive
             fields, ON/OFF onset channels, and learning by telling a real
             continuation of this sound from an impostor. Read out by a single
             learned layer it matches a hand-designed Gabor bank running in the
             same architecture (87.6% vs 85.6%) and far exceeds the same
             architecture with random fields (58.4%).

world/ -- what is out there, and what happens next
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
concepts     Concept assemblies, word grounding, and prediction. (Was
             world_model.py -- renamed to end the worldmodel/world_model clash.)
predictive   Predictive world model: transitions + successor representation +
             action-conditioned dynamics + latent belief (plans, look-ahead).
             (Was worldmodel.py.)
objects      Object-centric causal world graph: objects with properties and
             relations; causal rules learned per PROPERTY so they generalise to
             objects never seen; effect cascades and counterfactuals.
grounding    vision -> objects: visual features discovered from CAUSAL
             prediction error alone, never from property labels.
loop         The closed cognitive loop, actually running: predict -> sense ->
             prediction error -> correct, in a discrete OR continuous world.
continuous   A continuous predictive world model that beats the stay baseline
             the discrete loop lost to.
topdown      Top-down prediction (Rao & Ballard), staged growth, and looking
             driven by prediction error.

memory/ -- storing, replaying, reconstructing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
assembly     Spiking episodic memory: store a pattern, complete it from a cue.
rhythm       Theta-gamma working memory and its capacity (load ~5-7).
relational   Relational memory over bound pairs.
psyche       Pallium-like associative cortex + a mental space that remembers,
             reconstructs a percept from a cross-modal cue, and imagines.
development  Developmental learning (competence-gated critical periods) and
             dream consolidation (prioritised replay: +13% with no new data).

cognition/ -- the higher faculties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pallium      The pallium as a distributed predictive relational architecture:
             hierarchy + recurrent settling + top-down prediction + action and
             relation operators (transitive spatial reasoning).
selfmodel    The self: body model and its limits, learned motor capability,
             interoception, and agency (efference copy vs. actual sensation).
analogy      Analogy and relations-between-relations (Hebbian relation maps).
reasoning    A semantic network over learned concepts.
discovery    Unsupervised discovery of concepts in an unlabelled stream.
dopamine     Reward prediction error + eligibility traces -> bandit and
             delayed-reward tasks.
attention    Region gains and top-down current.
imagination  Offline replay current -- the brain driving itself.
multimodal   Association area binding vision + hearing (cross-modal recall).

learning/ -- teaching, curricula, self-organisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
teacher      Inductive (imprinting) learning.
curriculum   The first concepts taught to the brain.
curriculum_advanced  Harder lessons: counting, classification, opposites.
knowledge    A foundation-knowledge brain.
language     Word grounding.
builder      Factory for a ready-to-teach default brain.
selforganize A visual cortex that DISCOVERS its receptive fields instead of
             being given them, and the same development for hearing.
tagging      Synaptic tagging and capture: does a decaying calcium deposit
             beside the weight really hold long-term correlations better?

minds/ -- whole assembled systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mind         Perception grounded into a world model + imagination + attention.
unified      One brain: perception + attention + comprehension + pallium memory
             + predictive world model + objects + self + sleep, end to end.
integrated   A million-neuron multi-area substrate + a sensory bridge.
endtoend     The whole chain measured only at its far end: spikes -> shared
             code -> concept, with the compounding loss stated.
perceptloop  The whole chain joined: pixels -> V1 spikes -> V2 fragments ->
             object file -> properties -> affordances -> causal model ->
             imagination; plus active perception ("I need another look").

viz/ -- watch it think
~~~~~~~~~~~~~~~~~~~~~~
visualizer   Live graph animation of the activation path, input -> output.
console      Mission control: a live graph of the whole integrated brain with a
             management dashboard (sensors, modalities, percept, vitals).
             Run it with ``python -m neurobrain.viz.console``.
dashboard    ORPHAN -- superseded by console; nothing imports it.

tools/ -- drive it, diagnose it, measure it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
runtime      The threaded live tick loop + homeostasis.
mapdebug     Why did a self-organizing map collapse? Input diversity, winner
             concentration, the neighbourhood, dynamic range, residual error.
bigbrain     Scale and capacity curves on a large substrate.
"""

from .core.neuron import (Neuron, NeuronType, Population, NEURON_TYPES, get_type,
                     type_id)
from .core.synapse import STDPConfig, SynapseBundle, TagConfig
from .core.region import (Region, ROLE_SENSORY, ROLE_ASSOCIATION, ROLE_MOTOR,
                     ROLE_MEMORY)
from .world.concepts import Concept, WorldModel
from .core.brain import Brain, SpikeFrame
from .sensing.sensors import (Encoder, TextEncoder, ImageEncoder, SoundEncoder,
                      VectorEncoder)
from .learning.teacher import Teacher, Lesson
from .learning.curriculum import teach_starter_curriculum, CurriculumReport
from .learning.curriculum_advanced import (teach_advanced_curriculum, classify,
                                  AdvancedReport)
from .learning.language import LanguageModel
from .learning.builder import (build_default_brain, BrainSpec,
                      build_large_brain, LargeBrainSpec,
                      build_comprehensive_substrate, ComprehensiveSpec,
                      build_knowledge_brain)
from .learning.knowledge import teach_foundation, FoundationReport
from .tools.runtime import BrainRuntime
from .cognition.imagination import Imagination
from .cognition.attention import Attention
from .cognition.discovery import ConceptDiscovery, make_clustered_data
from .cognition.reasoning import SemanticNetwork
from .memory.relational import RelationalMemory
from .core.biophysical import BiophysicalCircuit, nmda_mg_block
from .vision.unified_eye import UnifiedEye, Percept, Tag
from .vision.ventral import (SpikingConvLayer, ComplexCellLayer, ITLayer, SpikingPool,
                     VisionHierarchy, VentralStream, build_ventral_stream,
                     locate_template,
                     oriented_edges, corner_images, curve_images, shape_images,
                     hard_shape_images, two_object_image, SHAPE_CLASSES,
                     gabor_kernel, preferred_orientation, orientation_selectivity,
                     phase_invariance, composition_order,
                     TraceInvarianceLayer, transformation_sequence)
from .audition.audio import (Cochleagram, AuditoryStream, build_auditory_stream,
                    divisive_normalization,
                    tone, chirp, harmonic_tone, noise_burst, am_tone,
                    sound_dataset, SOUND_CLASSES)
from .cognition.multimodal import (AssociationArea, MultisensoryBrain,
                        build_multisensory_brain)
from .minds.mind import Mind, build_mind
from .core.persistence import save_brain, load_brain
from .sensing.realworld import (load_mnist, load_fashion_mnist, FASHION_CLASSES,
                       contrast_normalise, GrowingCategoryMap,
                       DigitRecognizer, build_digit_recognizer)
from .memory.psyche import (AssociativeCortex, MentalSpace, ReconstructiveMind,
                    build_reconstructive_mind)
from .cognition.analogy import RelationalMind
from .world.predictive import PredictiveWorldModel, GridWorld
from .core.dendrite import (DendriticTree, build_cortical_column,
                      dendritic_density_report)
from .world.objects import (ObjectConcept, Relation, CausalWorldGraph,
                     build_causal_world, causal_generalisation_score,
                     TRAIN_OBJECTS, NOVEL_OBJECTS)
from .cognition.pallium import (PalliumLayer, PredictivePallium, build_predictive_pallium)
from .cognition.selfmodel import (BodyModel, Interoception, MotorCapability, AgencyModel,
                       SelfModel, build_self_model)
from .memory.development import (Episode, EpisodicBuffer, CriticalPeriod, Stage,
                         DevelopmentalProgram, SleepConsolidator, SemanticCortex,
                         sleep_benefit_experiment, build_developmental_program)
from .workspace import (GlobalWorkspace, GroundedConcept, workspace_binding_demo)
from .tools.bigbrain import (CorticalMemory, BigBrainReport, run_cognition_on_substrate,
                      substrate_capacity_curve)
from .world.loop import (SensoryWorld, ContinuousWorld, CognitiveLoop, LoopTrace,
                  closed_loop_experiment)
from .vision.spikingvision import (SpikingRetina, SpikingFeatureLayer, SpikingCategoryMap,
                           SpikingDigitBrain, build_spiking_digit_brain,
                           spiking_vs_rate_comparison)
from .world.grounding import (CausalFeatureLearner, GroundedCausalMind, render_object,
                       build_grounded_causal_mind, grounded_vs_handtyped)
from .vision.spikinghierarchy import (SpikingLayer, SpikingVentralHierarchy,
                              build_spiking_hierarchy, active_causal_discovery)
from .world.continuous import (ContinuousPredictor, decode_position,
                        continuous_prediction_experiment)
from .minds.endtoend import EndToEndBrain, build_end_to_end
from .vision.widev1 import (WideV1, WideV1Report, WideDigitBrain, build_wide_digit_brain,
                    build_wide_v1_report, width_ablation, window_ablation,
                    motion_experiment, architecture_decomposition,
                    grown_readout, drift_sequence, static_middle, DIRECTIONS)
from .learning.tagging import synaptic_tagging_experiment
from .sensing.streams import (Scene, build_scene, SaccadicEye, Fixation, ContinuousEar,
                     AuditoryBelt, Soundscape, SoundEvent, build_soundscape,
                     StreamingBrain, StreamReport, streaming_experiment,
                     onset_scores, MovingObject, MovingScene,
                     smooth_pursuit_experiment)
from .viz.console import BrainConsole, serve_console
from .learning.selforganize import (randomise_filters, develop_v1, develop_a1,
                          develop_by_prediction, grow_dendrites,
                          dendritic_footprints,
                          orientation_selectivity, filter_diversity,
                          spectrotemporal_tuning, self_organized_vs_designed,
                          self_organized_audio, SelfOrganizeReport)
from .vision.v2binding import (V1Activity, V2Binding, VentralV1V2, v1_parts,
                       preferred_orientations, build_v1_v2)
from .minds.perceptloop import (ObjectFile, PerceptualObjectMind, LoopReport,
                         full_loop_experiment)
from .tools.mapdebug import (MapDiagnosis, diagnose, diagnose_vision_vs_hearing,
                      participation_ratio, patch_statistics, map_diversity,
                      winner_statistics)
from .vision.composite import (SHAPES, draw_shape, overlapping_pair, composite_dataset,
                       part_overlap_report, LocalV2, CompositeReport,
                       composite_binding_experiment)
from .audition.auditorycortex import (PredictiveA1, A1Report, on_off_channels,
                            spectrotemporal_context, predictive_a1_experiment,
                            LatentPredictiveA1, PredictiveA2,
                            ContrastivePredictiveA1, TemporalPool, a1_states,
                            spectrotemporal_gabor_bank, MultiScaleA1,
                            linear_probe, SequenceContrastiveA2,
                            STDPContrastiveA1, InvariantContrastiveA1,
                            nuisance_transform, adapted_code,
                            vocal_tract_warp, speaker_normalize)
from .world.topdown import (PredictiveStack, GrowthStage, GrowthSchedule,
                     ActiveLooker, predictive_coding_experiment,
                     staged_vs_simultaneous, error_driven_looking_experiment)
from .memory.assembly import (SpikingEpisodicMemory, AssemblyReport,
                       assembly_experiment, random_episodes)
from .cognition.dopamine import (EligibilityTrace, DopaminergicModulator,
                       RewardPredictionError, DelayedRewardTask,
                       DopaminergicActionLoop, BanditTask,
                       compare_rules)
from .memory.rhythm import (OscillatoryWorkingMemory, RhythmReport,
                     working_memory_experiment)
from .core.backend import available_devices, best_device, compare_backends
from .minds.integrated import (IntegratedBrain, integration_experiment,
                         SensoryBridge)
from .minds.unified import UnifiedMind, build_unified_mind, pallium_capacity_curve
from .viz.visualizer import NetworkVisualizer

__version__ = "0.40.0"

__all__ = [
    "Neuron", "NeuronType", "Population", "NEURON_TYPES", "get_type", "type_id",
    "STDPConfig", "SynapseBundle", "TagConfig",
    "Region", "ROLE_SENSORY", "ROLE_ASSOCIATION", "ROLE_MOTOR", "ROLE_MEMORY",
    "Concept", "WorldModel",
    "Brain", "SpikeFrame",
    "Encoder", "TextEncoder", "ImageEncoder", "SoundEncoder", "VectorEncoder",
    "Teacher", "Lesson",
    "teach_starter_curriculum", "CurriculumReport",
    "teach_advanced_curriculum", "classify", "AdvancedReport",
    "LanguageModel",
    "build_default_brain", "BrainSpec",
    "build_large_brain", "LargeBrainSpec",
    "build_comprehensive_substrate", "ComprehensiveSpec",
    "build_knowledge_brain", "teach_foundation", "FoundationReport",
    "BrainRuntime", "Imagination", "Attention",
    "ConceptDiscovery", "make_clustered_data", "SemanticNetwork",
    "RelationalMemory", "BiophysicalCircuit", "nmda_mg_block",
    "SpikingConvLayer", "ComplexCellLayer", "ITLayer", "SpikingPool",
    "VisionHierarchy", "VentralStream", "build_ventral_stream", "oriented_edges",
    "corner_images", "curve_images", "shape_images", "hard_shape_images",
    "two_object_image", "SHAPE_CLASSES", "gabor_kernel", "preferred_orientation",
    "orientation_selectivity", "phase_invariance", "composition_order",
    "Cochleagram", "AuditoryStream", "build_auditory_stream", "tone", "chirp",
    "harmonic_tone", "noise_burst", "am_tone", "sound_dataset", "SOUND_CLASSES",
    "divisive_normalization",
    "AssociationArea", "MultisensoryBrain", "build_multisensory_brain",
    "Mind", "build_mind", "save_brain", "load_brain",
    "load_mnist", "contrast_normalise", "GrowingCategoryMap",
    "DigitRecognizer", "build_digit_recognizer",
    "AssociativeCortex", "MentalSpace", "ReconstructiveMind",
    "build_reconstructive_mind", "RelationalMind",
    "PredictiveWorldModel", "GridWorld",
    "DendriticTree", "build_cortical_column", "dendritic_density_report",
    "ObjectConcept", "Relation", "CausalWorldGraph", "build_causal_world",
    "causal_generalisation_score", "TRAIN_OBJECTS", "NOVEL_OBJECTS",
    "PalliumLayer", "PredictivePallium", "build_predictive_pallium",
    "BodyModel", "Interoception", "MotorCapability", "AgencyModel", "SelfModel",
    "build_self_model",
    "Episode", "EpisodicBuffer", "CriticalPeriod", "Stage",
    "DevelopmentalProgram", "SleepConsolidator", "SemanticCortex",
    "sleep_benefit_experiment", "build_developmental_program",
    "GlobalWorkspace", "GroundedConcept", "workspace_binding_demo",
    "SensoryWorld", "ContinuousWorld", "CognitiveLoop", "LoopTrace",
    "closed_loop_experiment",
    "SpikingRetina", "SpikingFeatureLayer", "SpikingCategoryMap",
    "SpikingDigitBrain", "build_spiking_digit_brain", "spiking_vs_rate_comparison",
    "CausalFeatureLearner", "GroundedCausalMind", "render_object",
    "build_grounded_causal_mind", "grounded_vs_handtyped",
    "SpikingLayer", "SpikingVentralHierarchy", "build_spiking_hierarchy",
    "active_causal_discovery", "ContinuousPredictor", "decode_position",
    "continuous_prediction_experiment", "EndToEndBrain", "build_end_to_end",
    "WideV1", "WideV1Report", "WideDigitBrain", "build_wide_digit_brain",
    "build_wide_v1_report", "width_ablation", "window_ablation",
    "motion_experiment", "architecture_decomposition", "grown_readout",
    "drift_sequence", "static_middle", "DIRECTIONS",
    "synaptic_tagging_experiment",
    "Scene", "build_scene", "SaccadicEye", "Fixation", "ContinuousEar",
    "AuditoryBelt", "Soundscape", "SoundEvent", "build_soundscape",
    "StreamingBrain", "StreamReport", "streaming_experiment", "onset_scores",
    "MovingObject", "MovingScene", "smooth_pursuit_experiment",
    "BrainConsole", "serve_console",
    "randomise_filters", "develop_v1", "develop_a1", "orientation_selectivity",
    "develop_by_prediction", "grow_dendrites", "dendritic_footprints",
    "filter_diversity", "spectrotemporal_tuning", "self_organized_vs_designed",
    "self_organized_audio", "SelfOrganizeReport",
    "V1Activity", "V2Binding", "VentralV1V2", "v1_parts",
    "preferred_orientations", "build_v1_v2",
    "ObjectFile", "PerceptualObjectMind", "LoopReport", "full_loop_experiment",
    "MapDiagnosis", "diagnose", "diagnose_vision_vs_hearing",
    "participation_ratio", "patch_statistics", "map_diversity",
    "winner_statistics",
    "SHAPES", "draw_shape", "overlapping_pair", "composite_dataset",
    "part_overlap_report", "LocalV2", "CompositeReport",
    "composite_binding_experiment",
    "PredictiveA1", "A1Report", "on_off_channels", "spectrotemporal_context",
    "predictive_a1_experiment",
    "PredictiveStack", "GrowthStage", "GrowthSchedule", "ActiveLooker",
    "predictive_coding_experiment", "staged_vs_simultaneous",
    "error_driven_looking_experiment",
    "CorticalMemory", "BigBrainReport", "run_cognition_on_substrate",
    "substrate_capacity_curve", "load_fashion_mnist", "FASHION_CLASSES",
    "TraceInvarianceLayer", "transformation_sequence",
    "UnifiedMind", "build_unified_mind", "pallium_capacity_curve",
    "NetworkVisualizer",
]
