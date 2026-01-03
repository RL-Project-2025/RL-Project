# Water Management - RL Algorithm Testing

Reinforcement Learning project using gym4ReaL.

## Current Task (Week 14)

**Main Goal**: Run all models, complete video and report.

### Model Runs - DEADLINE: 2 DAYS

| Person | Status | Task |
|--------|--------|------|
| Kavish | [x] Done | All runs complete |
| Kamal | [x] Done | All runs complete |
| Amy | [ x ] Pending | Clone code, run on HEX (limit CPU) |
| Reece | [ x ] Pending | Clone code, run on HEX |
| Hulusi | [ x ] Pending | No Norm + SMA for both algorithms |
| Radha | [ X ] Done | Gave TRPO changes to Hulusi |

**Run Configurations**: EMA/SMA × Norm/NoNorm (all combinations)

**Output Location**: Final `.pt`/`.zip` files go in `EMA+Normalisation/`

---

## Video Presentation

**Format**: 4 minutes total, ~40 seconds per person

**TO DO**: Draft your own sections, run it through Reece & Kamal, and start recording if they approve.

### Sections

1. **Introduction to the Environment** - Kavish
   - WDSEnv overview, gym4ReaL framework
   - Paper context and problem statement

2. **Environment Modifications** - Reece
   - Problems encountered
   - Fixes implemented (RewardScalingWrapper, Normalisation)
   - Currently existing issues

3. **Introduction to Models** - Amy
   - Main models and variations
   - SB3 baseline usage

4. **Model Modifications** - Hulusi
   - Changes from the paper
   - Optimisations from SB3
   - Network designs
   - Parameters used

5. **Model Performance** - Kamal
   - Avg episode reward (20 episodes)
   - Network resilience (1 episode)
   - Loss curves
   - Additional graphs

6. **Results & Evaluation** - Radha
   - Summary / overall story
   - Future work
   - Challenges

---

## Report

| Section | Status | Assigned |
|---------|--------|----------|
| 1-2 | [x] Nearly done | - |
| 3 (Algorithm writeups) | [ ] Pending | Each person for their algorithm |
| 4 (Model comparison) | [ ] Pending | Hulusi + Radha (after .zip/.pt files uploaded) |
| 5-6 | [ ] Pending | After Section 4 |

#### Footnote
- **Gang**: Final Stretch
