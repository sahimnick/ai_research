# Dormant code -- generated, do not edit by hand

Regenerate with `python3 docs/diagrams/make_diagrams.py`.

A class counts as **used** if its name appears in any .py or .md outside
the file that defines it. Being exported from `neurobrain/__init__.py`
does **not** count -- exporting a class is not using it, and this project
exports almost everything.

**81 modules, 166 classes, 82 never used outside their own file.**

| subpackage | classes | unused | share |
|---|---|---|---|
| `cognition` | 21 | **13** | 62% |
| `world` | 18 | **9** | 50% |
| `memory` | 15 | **9** | 60% |
| `learning` | 11 | **8** | 73% |
| `minds` | 10 | **8** | 80% |
| `audition` | 12 | **7** | 58% |
| `viz` | 9 | **7** | 78% |
| `vision` | 27 | **7** | 26% |
| `sensing` | 24 | **5** | 21% |
| `core` | 13 | **5** | 38% |
| `tools` | 4 | **3** | 75% |
| `workspace` | 2 | **1** | 50% |

## Every class with no user outside its own file

| class | module | exported |
|---|---|---|
| `A1Report` | `neurobrain.audition.auditorycortex` | yes |
| `InvariantContrastiveA1` | `neurobrain.audition.auditorycortex` | yes |
| `LatentPredictiveA1` | `neurobrain.audition.auditorycortex` | yes |
| `PredictiveA2` | `neurobrain.audition.auditorycortex` | yes |
| `STDPContrastiveA1` | `neurobrain.audition.auditorycortex` | yes |
| `SequenceContrastiveA2` | `neurobrain.audition.auditorycortex` | yes |
| `TemporalPool` | `neurobrain.audition.auditorycortex` | yes |
| `ConceptDiscovery` | `neurobrain.cognition.discovery` | yes |
| `BanditTask` | `neurobrain.cognition.dopamine` | yes |
| `DelayedRewardTask` | `neurobrain.cognition.dopamine` | yes |
| `DopaminergicActionLoop` | `neurobrain.cognition.dopamine` | yes |
| `DopaminergicModulator` | `neurobrain.cognition.dopamine` | yes |
| `EligibilityTrace` | `neurobrain.cognition.dopamine` | yes |
| `RewardPredictionError` | `neurobrain.cognition.dopamine` | yes |
| `PredictivePallium` | `neurobrain.cognition.pallium` | yes |
| `SemanticNetwork` | `neurobrain.cognition.reasoning` | yes |
| `AgencyModel` | `neurobrain.cognition.selfmodel` | yes |
| `BodyModel` | `neurobrain.cognition.selfmodel` | yes |
| `Interoception` | `neurobrain.cognition.selfmodel` | yes |
| `MotorCapability` | `neurobrain.cognition.selfmodel` | yes |
| `BiophysicalCircuit` | `neurobrain.core.biophysical` | yes |
| `DendriticTree` | `neurobrain.core.dendrite` | yes |
| `Neuron` | `neurobrain.core.neuron` | yes |
| `NeuronType` | `neurobrain.core.neuron` | yes |
| `_TypeView` | `neurobrain.core.neuron` | no |
| `BrainSpec` | `neurobrain.learning.builder` | yes |
| `ComprehensiveSpec` | `neurobrain.learning.builder` | yes |
| `LargeBrainSpec` | `neurobrain.learning.builder` | yes |
| `CurriculumReport` | `neurobrain.learning.curriculum` | yes |
| `AdvancedReport` | `neurobrain.learning.curriculum_advanced` | yes |
| `FoundationReport` | `neurobrain.learning.knowledge` | yes |
| `SelfOrganizeReport` | `neurobrain.learning.selforganize` | yes |
| `TagRun` | `neurobrain.learning.tagging` | no |
| `AssemblyReport` | `neurobrain.memory.assembly` | yes |
| `CriticalPeriod` | `neurobrain.memory.development` | yes |
| `DevelopmentalProgram` | `neurobrain.memory.development` | yes |
| `Episode` | `neurobrain.memory.development` | yes |
| `SemanticCortex` | `neurobrain.memory.development` | yes |
| `ReconstructiveMind` | `neurobrain.memory.psyche` | yes |
| `RelationalMemory` | `neurobrain.memory.relational` | yes |
| `OscillatoryWorkingMemory` | `neurobrain.memory.rhythm` | yes |
| `RhythmReport` | `neurobrain.memory.rhythm` | yes |
| `EndToEndBrain` | `neurobrain.minds.endtoend` | yes |
| `AreaReport` | `neurobrain.minds.integrated` | no |
| `IntegratedBrain` | `neurobrain.minds.integrated` | yes |
| `SensoryBridge` | `neurobrain.minds.integrated` | yes |
| `LoopReport` | `neurobrain.minds.perceptloop` | yes |
| `ObjectFile` | `neurobrain.minds.perceptloop` | yes |
| `PerceptualObjectMind` | `neurobrain.minds.perceptloop` | yes |
| `Comprehension` | `neurobrain.minds.unified` | no |
| `Encoder` | `neurobrain.sensing.sensors` | yes |
| `VectorEncoder` | `neurobrain.sensing.sensors` | yes |
| `SoundEvent` | `neurobrain.sensing.streams` | yes |
| `Soundscape` | `neurobrain.sensing.streams` | yes |
| `StreamReport` | `neurobrain.sensing.streams` | yes |
| `BigBrainReport` | `neurobrain.tools.bigbrain` | yes |
| `CorticalMemory` | `neurobrain.tools.bigbrain` | yes |
| `MapDiagnosis` | `neurobrain.tools.mapdebug` | yes |
| `CompositeReport` | `neurobrain.vision.composite` | yes |
| `SpikingLayer` | `neurobrain.vision.spikinghierarchy` | yes |
| `SpikingDigitBrain` | `neurobrain.vision.spikingvision` | yes |
| `V1Activity` | `neurobrain.vision.v2binding` | yes |
| `ITLayer` | `neurobrain.vision.ventral` | yes |
| `WideDigitBrain` | `neurobrain.vision.widev1` | yes |
| `WideV1Report` | `neurobrain.vision.widev1` | yes |
| `BrainConsole` | `neurobrain.viz.console` | yes |
| `NodeState` | `neurobrain.viz.console` | no |
| `_NpJSON` | `neurobrain.viz.console` | no |
| `BrainService` | `neurobrain.viz.dashboard` | no |
| `Handler` | `neurobrain.viz.dashboard` | no |
| `EyeService` | `neurobrain.viz.eye_dashboard` | yes |
| `_Source` | `neurobrain.viz.eye_dashboard` | no |
| `GroundedConcept` | `neurobrain.workspace` | yes |
| `ContinuousPredictor` | `neurobrain.world.continuous` | yes |
| `GroundedCausalMind` | `neurobrain.world.grounding` | yes |
| `LoopTrace` | `neurobrain.world.loop` | yes |
| `ObjectConcept` | `neurobrain.world.objects` | yes |
| `Relation` | `neurobrain.world.objects` | yes |
| `GridWorld` | `neurobrain.world.predictive` | yes |
| `ActiveLooker` | `neurobrain.world.topdown` | yes |
| `GrowthSchedule` | `neurobrain.world.topdown` | yes |
| `GrowthStage` | `neurobrain.world.topdown` | yes |
