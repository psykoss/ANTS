# 🐜ANTS: Adversarial Neural Testing Swarm for Physical AI

**A closed-loop adversarial testing framework for discovering, characterizing, and reducing the robustness failures of embodied AI policies.**

## Abstract

Robotic policies trained through reinforcement learning can achieve strong performance under nominal simulation conditions while remaining vulnerable to distribution shifts encountered during deployment. Changes in terrain geometry, actuator dynamics, sensor noise, illumination, visual distractors, and external disturbances can expose failure modes that are poorly represented in conventional training and evaluation environments.

Ants (Adversarial Neural Testing Swarm) is a proposed agentic framework for systematically identifying these failure modes. Rather than treating environment generation as a data-supply problem, Ants formulates physical AI evaluation as an iterative adversarial search process: generate perturbations, execute a policy, measure task degradation, classify failures, and construct structured evidence for subsequent policy improvement.

Built around a MuJoCo-based simulation workflow, with Microduck and the BallKick task as an initial experimental setting, Ants combines perturbation generation, policy rollouts, multimodal failure analysis, and adaptive test selection. A planning agent prioritizes promising perturbation families, a code-generation agent configures experiments, and vision-language analysis helps associate observed failures with their environmental and dynamical causes.

The central research objective is to investigate whether adaptive adversarial testing can discover consequential policy failures more efficiently than fixed, manually designed test suites, and whether the resulting failure dataset can support measurable improvements in robustness.

**Research premise:** A policy should not be considered robust simply because it succeeds on its training distribution. Robustness should be evaluated by how systematically its failure boundaries can be discovered, measured, and reduced.

---

## 1. Problem Formulation

### 1.1 Distributional fragility in embodied policies

Let a robot policy be represented by

\[
\pi_\theta(a_t \mid o_t),
\]

where \(\theta\) denotes the learned parameters, \(o_t\) the observations, and \(a_t\) the resulting actions.

Policy performance depends not only on the learned controller but also on the environment in which it operates. We represent an experimental condition by

\[
\xi = (\xi_{\mathrm{terrain}},
\xi_{\mathrm{dynamics}},
\xi_{\mathrm{sensor}},
\xi_{\mathrm{visual}},
\xi_{\mathrm{disturbance}}).
\]

These variables describe perturbations to the physical and observational conditions experienced by the robot.

For a task-specific return \(R(\pi_\theta,\xi)\), nominal evaluation estimates performance under a reference distribution \(p_{\mathrm{train}}(\xi)\). Deployment, however, may expose the policy to conditions drawn from a different distribution \(p_{\mathrm{deploy}}(\xi)\).

The resulting robustness gap can be expressed as

\[
\Delta_{\mathrm{robust}} =
\mathbb{E}_{\xi\sim p_{\mathrm{train}}}
[R(\pi_\theta,\xi)]
-
\mathbb{E}_{\xi\sim p_{\mathrm{deploy}}}
[R(\pi_\theta,\xi)].
\]

A policy may therefore exhibit high nominal performance while retaining a substantial robustness gap.

Conventional testing often relies on a limited set of hand-selected scenarios or random perturbations. Such approaches can miss rare, interacting, or highly consequential conditions, particularly when simulation rollouts are computationally expensive.

### 1.2 Research gap

Ants targets three limitations in existing policy evaluation workflows:

- **Limited test diversity:** Manually specified scenarios may not cover the relevant failure surface.
- **Inefficient exploration:** Uniform random testing can spend substantial compute on benign conditions instead of informative failures.
- **Disconnected evaluation and training:** Failures are often reported as aggregate metrics or isolated demonstrations rather than reusable, structured examples for policy improvement.

Ants addresses these limitations by treating testing as an adaptive search problem and linking the discovery process to a structured failure dataset.

---

## 2. Research Objectives

The project investigates four questions:

1. **Failure discovery:** Can an adaptive adversarial testing strategy identify policy failures that are missed by nominal evaluation and fixed test suites?
2. **Search efficiency:** Can feedback-driven perturbation selection increase the number of meaningful failures discovered per simulation rollout?
3. **Failure characterization:** Can video, task metrics, and robot telemetry be combined to produce useful and reproducible failure labels?
4. **Robustness improvement:** Can retraining on discovered failure cases improve performance under held-out perturbations without materially degrading nominal task performance?

These objectives define the experimental program. They are hypotheses to evaluate rather than established properties of the current implementation.

---

## 3. System Architecture

Ants is organized as a closed-loop pipeline with six functional stages.

### 3.1 Adaptive test planning

A planning agent analyzes previous experiments, including failure rates, perturbation parameters, severity, and uncertainty. It selects the next perturbation family or region of the test space to investigate.

Candidate strategies include uncertainty-driven exploration, failure-rate maximization, coverage-based selection, and reward degradation.

The planner should operate on a constrained set of experiment parameters rather than unrestricted environment modifications. This makes experiments reproducible and keeps generated tests within defined simulation and safety limits.

### 3.2 Perturbation generation

A code-generation agent translates the selected test objective into validated configuration files compatible with the simulator's existing workflow.

Initial perturbation families include:

| Category | Example perturbations |
|---|---|
| Terrain | Friction changes, uneven surfaces, altered geometry |
| Actuator dynamics | Motor strength variation, latency, backlash, control noise |
| Proprioception | Measurement noise, bias, delayed observations |
| Visual conditions | Lighting changes, textures, distractors, reduced contrast |
| External disturbances | Impulses, contact disturbances, unexpected object motion |

For vision-dependent tasks such as BallKick, image generation can introduce controlled variations in appearance, background, and illumination. Physical dynamics and observation perturbations remain separate experimental factors so that their individual effects can be measured.

Every generated configuration is validated against parameter bounds and stored with a unique experiment identifier.

### 3.3 Simulation and policy execution

The simulation layer uses MuJoCo and the selected Microduck/mjlab workflow to instantiate perturbed environments and execute a fixed policy checkpoint.

For each rollout, the system records:

- Initial conditions and random seeds.
- Perturbation parameters and environment configuration.
- Policy checkpoint and simulator versions.
- Task rewards, completion status, and termination reason.
- Joint positions, velocities, control signals, and relevant contact telemetry.
- Video and observation traces where available.

The initial task is BallKick, where failure can be measured through task completion, ball-contact behavior, target accuracy, instability, and timeout conditions.

A key experimental distinction is whether the policy receives camera observations, proprioception, or both. Ants preserves the policy's intended observation interface while varying the conditions that generate those observations.

### 3.4 Visual pre-filtering

A visual perception module can estimate the visibility and segmentation of task-relevant objects, such as the ball and target.

A segmentation model such as SAM 3.1 may be used to estimate object visibility, apparent size, occlusion, or mask quality. These measurements can help prioritize visual conditions that are likely to affect task perception.

However, segmentation quality is only a proxy for policy difficulty. A scene with a clearly segmented object can still cause policy failure, while a visually degraded scene may remain easy for a policy with strong temporal or proprioceptive cues.

Consequently, the pre-filter is treated as an experimental ranking mechanism, not a definitive predictor of failure. Its value must be assessed against unfiltered rollouts and actual policy outcomes.

### 3.5 Failure detection and classification

After each rollout, the evaluation layer determines whether the task succeeded and associates the outcome with the observed trajectory.

A multimodal judge can assist with interpreting video and telemetry, producing candidate labels such as:

- **Fall:** Loss of balance or an unsuccessful recovery.
- **Missed kick:** Failure to make the required ball contact or achieve the intended outcome.
- **Instability:** Excessive oscillation, abnormal joint behavior, or repeated corrective actions.
- **Timeout:** Failure to complete the task within the allocated horizon.
- **Perception-related failure:** Behavior consistent with visual ambiguity or observation degradation.
- **Unclassified:** A failure that cannot be assigned a reliable category.

Where possible, labels are derived from deterministic task metrics and simulator signals first, with model-generated interpretation used as supporting evidence.

Each failure record includes the evidence used to assign its label and an uncertainty indicator. This separation reduces the risk of treating a language-model judgment as ground truth.

### 3.6 Structured failure dataset

Every informative rollout produces a versioned record containing:

- Policy and environment versions.
- Perturbation family and parameter values.
- Initial state, random seed, and rollout identifier.
- Success metrics and failure labels.
- Video references and telemetry traces.
- Failure severity and confidence, where defined.
- Reproduction instructions.

The resulting dataset is intended to support regression testing, failure analysis, and targeted policy retraining.

---

## 4. Closed-Loop Adversarial Testing

The defining component of Ants is the feedback loop between test generation and observed policy behavior.

Let \(q_t\) denote the distribution used to sample perturbations at iteration \(t\). After executing a batch of tests, the system obtains outcomes \(D_t\), which update the next sampling strategy:

\[
q_{t+1} = \mathcal{U}(q_t,D_t),
\]

where \(\mathcal{U}\) represents the test-selection update rule.

The objective is not simply to maximize the number of failures. An effective tester should discover reproducible, consequential failures while maintaining enough diversity to avoid repeatedly testing the same narrow region.

A practical iteration consists of:

1. Select a perturbation family using prior rollout evidence.
2. Generate and validate a batch of environment configurations.
3. Execute the policy under controlled initial conditions.
4. Measure task outcomes and collect telemetry.
5. Classify and rank the observed failures.
6. Update the perturbation search strategy.
7. Export reproducible cases for regression testing and retraining.
8. Evaluate the updated policy against both discovered and held-out cases.

This process supports a continuous cycle:

**Attack → Observe → Diagnose → Retrain → Re-attack.**

---

## 5. Experimental Methodology

Ants should initially be evaluated against fixed baselines to determine whether adaptive testing provides a measurable advantage.

### 5.1 Baselines

The proposed comparisons are:

- **Nominal evaluation:** The unperturbed task environment.
- **Random testing:** Perturbations sampled from predefined parameter distributions.
- **Fixed test suite:** A manually specified collection of representative scenarios.
- **Adaptive adversarial testing:** Perturbation selection informed by previous rollout outcomes.

All methods should receive comparable rollout budgets, policy checkpoints, parameter bounds, and evaluation conditions.

### 5.2 Evaluation metrics

| Metric | Purpose |
|---|---|
| Nominal success rate | Measures performance under reference conditions |
| Perturbed success rate | Measures task completion under distribution shifts |
| Failure discovery rate | Counts unique, reproducible failure modes found per rollout budget |
| Time to first failure | Measures how quickly testing identifies a policy weakness |
| Perturbation coverage | Measures exploration of the defined parameter space |
| Worst-case performance | Measures degradation in selected high-risk scenarios |
| Classification agreement | Measures agreement between automated labels and human-reviewed labels |
| Retraining improvement | Measures changes in robustness after incorporating failure cases |
| Regression rate | Measures whether improvements introduce new failures or reduce nominal performance |

A useful primary metric is the discovery curve: the cumulative number of unique, validated failure modes found as a function of rollout count or simulation time.

This enables a direct comparison between adaptive testing, random sampling, and fixed test suites under equal computational budgets.

### 5.3 Ablation studies

To determine which components contribute measurable value, the evaluation should compare:

- Adaptive versus random perturbation selection.
- Visual pre-filtering versus no pre-filtering.
- Telemetry-based classification versus multimodal classification.
- Single-factor perturbations versus controlled combinations of factors.
- Retraining with discovered failures versus retraining without them.

Experiments should use multiple random seeds, a held-out perturbation set, and confidence intervals where sample sizes permit.

### 5.4 Robustness improvement

The retraining stage uses selected failure cases to augment the policy's training distribution. The updated checkpoint is then evaluated on three distinct sets:

1. Previously discovered failure scenarios.
2. Unseen perturbations from the same perturbation families.
3. Held-out perturbation families or combinations.

This separation helps distinguish memorization of known scenarios from more general robustness improvements.

A successful result would demonstrate improved performance on held-out conditions while retaining acceptable nominal task performance.

---

## 6. Initial Demonstration

The initial demonstrator targets a Microduck-based BallKick workflow in MuJoCo/mjlab.

The demonstration is designed to show an end-to-end testing cycle:

1. Load a trained policy checkpoint.
2. Run the policy in the nominal environment.
3. Generate controlled visual or physical perturbations.
4. Identify a condition that degrades task performance.
5. Produce a reproducible failure record with video and telemetry.
6. Incorporate selected failure cases into a subsequent training run.
7. Re-evaluate the updated policy on the original failure and held-out scenarios.

The key output is not an isolated robot failure video. It is a reproducible link between a perturbation, an observed failure, a structured diagnosis, and a measurable change in policy performance.

Where a physical robot is available, selected simulation-discovered scenarios can be transferred to hardware under controlled conditions. This is a separate validation stage: simulation success alone does not establish real-world robustness.

---

## 7. Implementation Principles

Ants follows several design principles intended to keep the experimental system reliable.

**Reproducibility.** Every rollout is associated with a policy version, simulator configuration, perturbation parameters, and random seed.

**Separation of concerns.** Test planning, configuration generation, simulation, measurement, and interpretation are separate modules. Language models propose and interpret experiments; deterministic code validates configurations and computes objective metrics.

**Bounded adversarial search.** Perturbations remain within explicit limits. Invalid configurations are rejected before execution, and physical tests require additional hardware-specific safety controls.

**Evidence-based failure labels.** Automated interpretations are linked to measurable outcomes and supporting traces. Ambiguous cases remain unclassified until reviewed.

**Budget-aware evaluation.** Test selection accounts for simulation cost, discovery efficiency, and diversity rather than optimizing only for raw failure count.

**Hardware-aware validation.** Simulation findings are treated as hypotheses about real-robot behavior and are validated separately before being used to support deployment claims.

---

## 8. Scope and Limitations

The initial system focuses on discovering and characterizing failures in a defined robotic task, not proving formal safety or universal policy robustness.

Several limitations remain important:

- The quality of the search depends on the perturbation families and parameter bounds available to the tester.
- A simulator may not reproduce all real-world contact, actuator, sensing, and environmental effects.
- Visual segmentation and multimodal judgments can be unreliable and require validation.
- Discovered failures may be task-specific and may not generalize to other robots or policies.
- Retraining on adversarial cases can overfit to known scenarios or degrade nominal performance.
- Simulation-to-hardware transfer requires separate evaluation and safety procedures.

These limitations motivate the use of held-out scenarios, explicit baselines, independent validation, and reproducible experiment records.

---

## 9. Expected Contribution

Ants explores a shift from passive evaluation to active, adaptive testing of embodied intelligence.

The intended contribution is a practical methodology for connecting adversarial environment generation, policy evaluation, failure characterization, and retraining within one reproducible workflow.

Rather than optimizing for the quantity of collected physical AI data, Ants prioritizes the information value of each experiment: which conditions expose weaknesses, what evidence explains the observed failure, and whether incorporating that evidence makes the policy more robust.

The long-term direction is a reusable robustness-testing infrastructure for robot policies across simulators, tasks, and hardware platforms.

**Ants — Find the failure before the robot does.**

---

## Project Status

This repository is intended to develop and evaluate the proposed framework incrementally. The architecture and research questions described here define the target system; individual integrations, baseline comparisons, and robustness improvements should be considered validated only when supported by reproducible experiments.

Implementation details, supported tasks, evaluation results, and hardware validation status should be updated as the project progresses.
