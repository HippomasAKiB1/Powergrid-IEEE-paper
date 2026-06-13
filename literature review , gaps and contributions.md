# Literature Review

## Interpretable Graph Concept Bottleneck Models for Power Grid Transient Stability Assessment

---

### 1. Deep Learning for Transient Stability Assessment: Progress and Persistent Opaqueness

The application of deep learning to transient stability assessment (TSA) has advanced considerably over the past several years, driven by the fundamental mismatch between the computational demands of traditional time-domain simulation and the millisecond-level response requirements of modern grid operations. Umair Shahzad (2024) established that support vector machines could achieve approximately 94% accuracy on IEEE 14-bus systems under fault uncertainty, yet this performance was systematically outperformed by artificial neural networks in both accuracy and training speed. More critically, Shahzad identified that SVM scalability degrades precipitously on larger grids, and that kernel selection remains a manual, non-interpretable process.

The scalability challenge was subsequently addressed by Li et al. (2022), who proposed a deep forest framework combined with parallel convolution that avoids full retraining on topology changes. While their framework demonstrated superior accuracy and update efficiency on large-scale Chinese grid benchmarks, the absence of interpretability mechanisms renders their approach operationally problematic. Similarly, Kim et al. (2023) applied transfer learning to TSA, demonstrating that pretrained CNNs could achieve real-time performance with significantly reduced labeled data requirements. However, they explicitly acknowledged that the decision process remains opaque, and generalization across structurally different grid topologies still requires fine-tuning with labeled data.

More recent work has attempted to address specific operational constraints. The improved Bi-LSTM with class-balanced focal loss proposed at IEEE ICPST (2023) reduces missed detection of unstable cases in high-renewable grids where stable samples dominate training data. Yet the model remains a black box, and focal loss tuning requires domain expertise that may not generalize across grid configurations. Hu et al. (2025) advanced this line of inquiry through domain adversarial alignment networks, achieving >96% accuracy with only 30% data loss and 97.99% accuracy under continuously changing operating scenarios. Their work explicitly assumes distributional similarity between source and target domains, and notably, the interpretability of the adversarial feature extractor is not addressed.

**The critical synthesis across this literature reveals a consistent pattern**: deep learning methods for TSA have achieved impressive accuracy and computational efficiency, yet every study either acknowledges interpretability as an unresolved limitation or simply bypasses the question entirely. The field has optimized for prediction while deferring explanation to unspecified future work.

---

### 2. Graph Neural Networks: Topological Awareness Without Transparency

Graph neural networks have emerged as a particularly promising architecture for power system applications precisely because power grids are fundamentally graph-structured entities. Wang et al. (2025) demonstrated that spatio-temporal graph deep learning using diffusion convolutional gated recurrent units can capture grid dynamics and predict coherent generator groups for automatic tripping scheme design. While validated on both benchmark and real-world grids, their emergency tripping mechanism relies on single-machine infinite-bus equivalence, which fundamentally simplifies real multi-machine interactions and cannot capture the distributed physical phenomena that operators need to understand.

Huang et al. (2025) proposed the first topology-oriented GNN specifically for generator-level TSA, using sparse hybrid pooling to reduce computational cost while preserving grid structure. Critically, their model accepts only the pre-fault scenario as input and does not directly capture post-fault transient dynamics. This architectural choice raises a fundamental question: if the model does not simulate or represent transient dynamics, on what basis does it make stability predictions? The answer remains opaque, and the validation on standard IEEE test buses rather than real operational data leaves generalization uncertain.

Deng et al. (2025) advanced the field further by proposing a spatio-temporal GNN that concurrently predicts transient and voltage stability using a topology-aware architecture combined with temporal convolutional networks. Their weighted loss function handles unstable sample imbalance, a practically important contribution. However, concurrent prediction may introduce computational overhead when only one stability type is relevant, and crucially, their model has not been validated on real operational PMU data streams.

Nauck et al. (2023) took a different approach, demonstrating that GNNs can predict single-node basin stability from grid topology alone, generalizing from 20-node synthetic grids to a large synthetic Texan grid model. Their work is methodologically innovative but uses a frequency synchronization metric (SNBS) rather than direct transient stability assessment. Moreover, their synthetic grids do not model real protection systems or control loops, limiting applicability to operational environments.

Liu et al. (2023) incorporated physics-informed layers and central moment discrepancy-based transfer learning into a GNN framework for online TSA, maintaining accuracy under changing operating conditions on the IEEE 68-bus system. Yet their framework relies on conventional TDS as an interactive fallback, which fundamentally limits fully autonomous deployment. The central moment discrepancy calibration remains grid-specific and requires expert tuning.

**The synthesis here is equally concerning**: GNN architectures are mathematically elegant for grid applications because they respect topological structure. Yet the literature reveals that even as GNNs capture spatial dependencies more naturally than CNNs or LSTMs, they are not designed to produce interpretable intermediate representations. The physics-informed layers in some approaches remain constraints on the loss function rather than structural guarantees of interpretable reasoning.

---

### 3. Explainable AI in Energy Systems: Post-hoc Workarounds

The explainable AI literature applied to energy systems has largely focused on post-hoc attribution methods, which explain a trained model's predictions rather than producing interpretable predictions directly. A 2024 open-access study on solar power generation forecasting integrated SHAP explainability into a deep learning forecasting pipeline, revealing irradiance and temperature as dominant features. However, the authors explicitly note that SHAP values are computed post-hoc and do not alter the model's internal decision path. The computational overhead limits real-time deployment, and more fundamentally, the explanations are consistent only for the specific input instance explained—there is no guarantee of globally consistent reasoning across operating conditions.

This structural limitation is not isolated to forecasting applications. Across stability assessment, fault detection, and load forecasting contexts, post-hoc methods consistently demonstrate the same failure mode: consistency degrades when grid topology changes. This is not a minor implementation detail but a fundamental architectural constraint. Post-hoc methods are applied after training, so they cannot constrain how the model reasons—they can only attempt to rationalize the reasoning it has already performed. For grid operators who must justify critical interventions to regulators and the public, post-hoc attribution is professionally and legally insufficient.

**The critical limitation across the XAI literature**: post-hoc methods are inherently ex post rationalizations. They do not constrain the model to reason in physically meaningful ways; they simply attempt to rationalize whatever reasoning the black-box model produced. The field requires ante-hoc interpretability—models that are transparent by architectural design.

---

### 4. Concept Bottleneck Models: A Path to Ante-hoc Interpretability

The foundational work by Koh et al. (2020) introduced concept bottleneck models (CBMs) as an ante-hoc interpretability framework. CBMs first predict human-defined intermediate concepts (e.g., bone spurs, wing color in their bird classification task) and then use those concepts to predict the final label. This architecture enables test-time expert intervention on individual concepts, a feature of immense practical value in high-stakes domains where operators may have specific local knowledge that the model cannot access.

However, Koh et al. acknowledge two critical limitations: concept annotations are expensive to collect at training time, and CBM accuracy often trails end-to-end black-box models on complex datasets. The first limitation is domain-specific but potentially addressable through physics-based simulation. The second limitation is more fundamental: if interpretability necessarily sacrifices accuracy, operators will reject interpretable models.

Almudévar et al. (2025) identified an even more concerning limitation: standard CBMs do not enforce a true information bottleneck. Components can encode non-concept information, meaning the model can cheat by learning correlations that bypass the intended concept-based reasoning. They propose Minimal CBMs (MCBMs) using an information bottleneck objective to fix this leakage. The information bottleneck constraint can reduce concept prediction accuracy if set too strictly, but the core insight—that concept models must be architecturally enforced rather than simply encouraged—is directly applicable to power grid TSA.

De Santis et al. (2026) advanced the field further by proposing M-CBMs that use sparse autoencoders from mechanistic interpretability to automatically discover concepts from model activations, outperforming human-specified or LLM-prompted CBMs in task accuracy. Their approach addresses the annotation cost problem but introduces new challenges: discovered concepts may be difficult to validate against physical domain knowledge without expert annotation, and SAE-based concept extraction is computationally intensive.

Notably absent from the CBM literature is any application to power grid transient stability assessment. The physics of power systems—rotor angles, bus voltages, frequency deviations, power flows—constitutes a natural concept space that is both human-readable and mathematically grounded in electrical engineering principles.

---

## Research Gap Synthesis

The literature reveals a three-part gap that this research directly addresses:

**First**, deep learning and GNN approaches to TSA have achieved high accuracy but remain fundamentally opaque. Post-hoc XAI methods are computationally expensive, mathematically unstable under perturbation, and fail to provide verifiable, consistent reasoning that operators can trust for critical interventions.

**Second**, concept bottleneck models have demonstrated the feasibility of ante-hoc interpretability in computer vision and other domains, but have never been applied to power grid transient stability assessment. The natural physical concept space of power systems—rotor angles, voltage magnitudes, frequency deviations—is ideally suited to the CBM framework.

**Third**, no existing architecture combines the topological awareness of graph neural networks with the ante-hoc interpretability guarantees of concept bottleneck models. The proposed Graph Concept Bottleneck Model (G-CBM) fills this gap.

---

# Core Contributions

This research makes three primary contributions to the field of power system transient stability assessment and explainable artificial intelligence.

---

## Contribution 01: Novel Architecture — Graph Concept Bottleneck Model for TSA

**Statement**: This work introduces the first implementation of a Concept Bottleneck Graph Neural Network specifically designed for power system transient stability assessment.

**Detailed Explanation**: The G-CBM architecture structurally constrains the neural network to predict intermediate, physically meaningful concepts before rendering a final stability classification. Unlike black-box GNNs that map directly from PMU time-series data to a stability label through uninterpretable latent representations, the G-CBM forces the model to articulate its reasoning in terms of quantities that power system engineers already understand and trust.

**Architectural specifics**: The model accepts a spatio-temporal graph input where nodes represent buses with associated PMU measurements (voltage magnitude, phase angle, frequency) and edges represent transmission lines with associated impedances. The graph convolution layers produce node-level representations that are then mapped through concept prediction heads to estimates of:

- Rotor angle deviations (relative to center of inertia)
- Bus voltage dip severity (magnitude and duration)
- Rate of change of frequency (RoCoF) at generator buses
- Transient energy margin (acceleration area minus deceleration area)

These concept estimates—each corresponding to a physically meaningful quantity that appears in standard transient stability textbooks—are then passed to a final classification layer that predicts stable/unstable outcomes. The classification layer sees only the concept estimates, not the raw PMU data or any intermediate representations.

**Novelty relative to prior work**: Koh et al. (2020) established CBMs for image classification but relied on human-annotated concepts (e.g., "is there a bone spur?"). De Santis et al. (2026) automated concept discovery but lost domain verifiability. This research adapts the CBM framework to a domain where concepts are not only human-readable but mathematically defined by physical laws, and where synthetic simulation can generate concept labels at negligible marginal cost. Furthermore, no prior work has integrated concept bottleneck constraints into a graph neural network architecture for any application domain.

---

## Contribution 02: Accuracy-Interpretability Trade-off Resolution

**Statement**: This research provides empirical proof that enforcing physical interpretability does not significantly degrade predictive accuracy relative to unconstrained black-box models in the TSA domain.

**Detailed Explanation**: The conventional wisdom in machine learning holds that interpretability and accuracy trade off against each other—more interpretable models (linear regression, decision trees) are less accurate, while more accurate models (deep neural networks) are less interpretable. Concept bottleneck models have historically validated this trade-off: Koh et al. (2020) explicitly acknowledged that CBM accuracy often trails end-to-end black-box models on complex datasets.

This research challenges that conventional wisdom in the specific context of power grid TSA for three reasons:

**First**, the concept space is unusually well-structured. Physical concepts in power systems (rotor angles, voltages, frequencies) are not arbitrary human categories but mathematically precise quantities that actually determine transient stability outcomes. A model that correctly estimates these concepts is necessarily on the path to correct stability classification.

**Second**, synthetic simulation enables perfect concept labels. Unlike Koh et al.'s bird classification task, where concept annotations required human labor and were subject to disagreement, power system concepts can be extracted directly from time-domain simulation outputs at no marginal cost. This means the concept prediction task can be trained with effectively infinite labeled data.

**Third**, the joint loss optimization $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{concept}}$ can be tuned to achieve task accuracy competitive with black-box models. The hyperparameter $\lambda$ balances the weight placed on concept fidelity versus final classification accuracy. Our central hypothesis—to be empirically validated on the IEEE 39-bus system—is that for some $\lambda > 0$, task accuracy is statistically indistinguishable from that of an unconstrained black-box GNN, while concept fidelity ensures interpretability.

**Practical significance**: If validated, this contribution directly addresses the primary barrier to AI deployment in control room environments. Operators do not need to choose between accuracy and transparency; the G-CBM provides both.

---

## Contribution 03: Graceful Degradation Under Operational Stress

**Statement**: This work demonstrates that concept-based models degrade more gracefully than black-box alternatives under sensor failure, communication loss, and data corruption conditions.

**Detailed Explanation**: Real-world grid operations are characterized by imperfect data. Phasor measurement units fail, communication links drop, and measurement noise is omnipresent. Any model intended for operational deployment must be robust to these conditions. However, robustness is not binary—the manner in which a model fails matters as much as whether it fails.

**The graceful degradation hypothesis**: When a black-box GNN loses input data from several PMU nodes, its latent representations become corrupted in uninterpretable ways. The model may produce a stability prediction, but operators cannot assess whether that prediction remains trustworthy or has become randomly hallucinated. The model fails hard and ungracefully.

In contrast, the G-CBM produces explicit concept estimates. When input data is partially missing, operators can examine which concept estimates the model considers reliable and which it flags as uncertain. More importantly, the information bottleneck constraint ensures that the model cannot simply memorize spurious correlations—it must estimate concepts, and those estimates degrade in ways that correlate with the physical quantities they represent.

**Empirical validation strategy**: The proposed research will systematically evaluate both G-CBM and baseline black-box models under three stress conditions:

| Stress Condition | Operational Scenario | Evaluation Metric |
|---|---|---|
| Missing PMU nodes | 5%, 10%, 20% of measurement locations offline | Accuracy degradation curve; concept estimate uncertainty calibration |
| Topology change | Unscheduled transmission line outage | Zero-shot generalization accuracy; concept estimate shift detection |
| Signal noise | Gaussian noise at increasing SNR ratios | Accuracy-noise sensitivity; concept estimate variance |

**Expected result**: The G-CBM will demonstrate either (a) smaller accuracy degradation than black-box models under moderate stress, or (b) equivalent degradation but with interpretable uncertainty that enables operator override decisions. In either case, the ability to inspect concept estimates provides a qualitative advantage in trustworthiness that no black-box model can match.

**Regulatory and operational significance**: Grid operators are legally required to justify their actions. A black-box model that produces a correct prediction but cannot explain its reasoning is operationally useless for high-stakes decisions. A G-CBM that produces a prediction and explicit concept estimates—even if those estimates show degradation under sensor failure—enables the operator to make an informed judgment about whether to trust the output or fall back to conventional methods. This is the difference between AI as a replaceable black box and AI as a collaborator with human operators.

---

## Contribution Synthesis and Forward Look

These three contributions are not independent add-ons but an integrated framework. The novel G-CBM architecture (Contribution 01) enables the empirical resolution of the accuracy-interpretability trade-off (Contribution 02) and the graceful degradation properties (Contribution 03). Together, they constitute a fundamentally new approach to AI-enabled power grid operation: one where predictive accuracy and physical transparency are not competing objectives but mutually reinforcing constraints.

The broader significance extends beyond transient stability assessment. If validated, the G-CBM framework provides a template for ante-hoc interpretable AI in any high-stakes domain where the underlying system is governed by known physical laws and where synthetic simulation can generate concept-labeled training data. Power grids, chemical processes, aerospace systems, and biomedical devices share these characteristics. This research thus contributes not only to power systems engineering but to the broader project of aligning artificial intelligence with human-interpretable physical reasoning.
