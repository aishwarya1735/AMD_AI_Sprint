# Group Relative Policy Optimization (GRPO) Research

## Overview
GRPO is a Reinforcement Learning (RL) algorithm specifically designed to improve model reasoning ability while significantly reducing computational requirements compared to traditional PPO.

## Key Advantages of GRPO

### 1. Computational Efficiency
- **50% reduction in compute requirements** compared to PPO used for ChatGPT
- Eliminates the need for a value model, reducing from 4 LLMs to 3 LLMs in the training pipeline
- Enables training reasoning models with much less memory (e.g., 1B parameter Llama 3.2 model with 16GB VRAM)
- Makes RL training accessible for "GPU poor" researchers

### 2. Memory Usage Comparison
| Method | 0.5B | 1B | 3B | 8B | 14B |
|--------|------|----|----|----|----|
| Fine Tune | 25 GB | 38 GB | 58 GB | OOM | OOM |
| Fine Tune + LoRA | 16 GB | 26 GB | 52 GB | OOM | OOM |
| Fine Tune + LoRA + Gradient Checkpointing | 15 GB | 24 GB | 41 GB | 72 GB | OOM |
| PEFT LoRA | 13 GB | 16 GB | 32 GB | 58 GB | 69 GB |

### 3. Simplified Architecture
- **PPO**: Policy Model + Reference Model + Reward Model + Value Model (4 LLMs)
- **GRPO**: Policy Model + Reference Model + Reward Model (3 LLMs)
- **GRPO-Zero**: Policy Model + Reference Model (2 LLMs, eliminates reward model)

## DeepSeek-R1 Training Pipeline

The complete training process alternates between SFT and GRPO:

1. **Supervised Fine Tuning (SFT)**
   - Cold start with high quality data
   - A couple thousand examples verified by humans

2. **Reinforcement Learning w/ GRPO**
   - Train model to have reasoning traces `<reasoning></reasoning>`
   - Deterministic rewards for formatting, consistency, and correctness

3. **Supervised Fine Tuning (SFT)**
   - Generate 800k Synthetic SFT data points and reject/filter
   - LLM As A Judge to filter incorrect responses

4. **Reinforcement Learning w/ GRPO**
   - Align the model to be helpful and harmless

## How GRPO Works

### Core Mechanism: Group Relative Advantages

Instead of generating one output per query, GRPO generates multiple outputs (o₁, o₂, ..., oG) for each input query.

### Mathematical Formula
The advantage calculation in GRPO is:

```
Â_{i,t} = r̃_i = (r_i - mean(r)) / std(r)
```

Where:
- `r_i` is the reward for output i
- `mean(r)` is the average reward across all outputs in the group
- `std(r)` is the standard deviation of rewards in the group

### Example Process
For a math question like "Mr. Curtis has 325 chickens on his farm where 28 are roosters and the rest are hens. Twenty hens do not lay eggs while the rest of the hens do. How many egg-laying hens does Mr. Curtis have on his farm?"

1. **Generate Multiple Outputs**: Model produces several reasoning traces, some correct (answer=277) and some incorrect (answer=305)

2. **Calculate Rewards**: Each output gets rewards for:
   - Formatting = 1.0
   - Answer correctness = 0.0 or 1.0
   - Consistency = 0.5

3. **Compute Group Advantages**: Use the formula above to normalize rewards relative to the group

### Python Implementation
```python
import numpy as np
advantages = [(r_i - np.mean(r)) / np.std(r) for r_i in r]
```

## Key Benefits for Reasoning Tasks

1. **Eliminates Value Model**: No need to train a separate value estimation model
2. **Group-Based Learning**: Learns from comparing multiple attempts at the same problem
3. **Deterministic Rewards**: Can use verifiable metrics for mathematical/logical problems
4. **Memory Efficient**: Significantly reduced VRAM requirements
5. **Stable Training**: Fewer moving parts compared to PPO

## Hardware Requirements
- Successfully demonstrated on 1B parameter models with 16GB VRAM
- Scales efficiently to larger models with appropriate hardware
- Compatible with LoRA and gradient checkpointing for further memory reduction

## Implementation Considerations
- Requires generating multiple outputs per input (typically 4-8 outputs)
- Reward functions should be deterministic and verifiable
- Works well with reasoning traces and step-by-step problem solving
- Can be combined with synthetic data generation for scaling



## Direct Preference Optimization (DPO) with Self-Training

### Overview
DPO is a simpler alternative to RLHF that directly optimizes language models using preference data without requiring a separate reward model. When combined with self-training, it creates a powerful approach for enhancing reasoning capabilities.

### Key Advantages of DPO over RLHF

1. **Simplified Training Pipeline**: Eliminates the need for a separate reward model
2. **More Stable Training**: Avoids the instability issues common in RLHF
3. **Cost Effective**: 40% faster training with 60% lower costs compared to RLHF
4. **Direct Optimization**: Uses preference data directly without reinforcement learning loops

### DPO-Augmented Self-Training Algorithm

The methodology combines traditional self-training with DPO in an iterative process:

#### Algorithm Steps:
1. **Warm-up Stage**: Fine-tune base model on labeled dataset
2. **Iterative Process**:
   - **DPO Step**: Generate preference dataset and train with DPO
   - **SFT Step**: Build pseudo-labeled dataset and train with supervised fine-tuning
3. **Repeat** until convergence or maximum iterations

#### DPO Dataset Generation:
- Generate multiple completions for each input
- Create preference pairs where x' ≺ H and y'_w ≻ y'_l (preferred vs rejected)
- Use DPO loss function to optimize model

#### Mathematical Foundation:
DPO loss function:
```
L_DPO = -log σ(r(y_w|x) - r(y_l|x))
```

Where r(·|x) = β log π(y|x)/π_ref(y|x) and β controls deviation from reference model.

### Benefits for Reasoning Tasks

1. **Enhanced Chain-of-Thought**: Improves step-by-step reasoning quality
2. **Diverse Reasoning Paths**: Encourages exploration of different solution approaches
3. **Self-Improvement**: Models learn from their own generated outputs
4. **Scalable**: Doesn't require large proprietary models for data generation

### Comparison with Other Methods

| Method | Reward Model Required | Training Stability | Computational Cost | Performance |
|--------|----------------------|-------------------|-------------------|-------------|
| RLHF | Yes | Moderate | High | High |
| DPO | No | High | Medium | High |
| GRPO | Optional | High | Low | High |
| SFT Only | No | High | Low | Medium |

### Implementation Considerations

1. **Preference Data Quality**: Critical for DPO effectiveness
2. **Hyperparameter β**: Controls how much the model can deviate from reference
3. **Iteration Balance**: Proper balance between DPO and SFT steps
4. **Evaluation Metrics**: Use verifiable correctness for mathematical reasoning

### Applications for Logical Reasoning

- **Truth-teller/Liar Problems**: DPO can help distinguish between consistent and inconsistent reasoning
- **Seating Arrangements**: Preference learning helps identify correct constraint satisfaction
- **Blood Relations**: Improves accuracy in multi-step relationship inference


## Curriculum Learning for Reasoning Tasks

### E2H (Easy to Hard) Reasoner

#### Overview
Curriculum learning schedules tasks from easy to hard, allowing LLMs to build reasoning skills gradually. The E2H Reasoner approach has shown significant improvements in reasoning capabilities for small LLMs (1.5B to 3B parameters).

#### Key Principles

1. **Progressive Difficulty**: Start with simple tasks and gradually increase complexity
2. **Appropriate Scheduling**: Fading out easy tasks is essential to prevent overfitting
3. **Task Decomposition**: Break down complex reasoning into manageable levels
4. **Theoretical Foundation**: Convergence guarantees within approximate policy iteration framework

#### Task Hierarchy
- **Trivial Tasks**: Basic logical operations
- **Easy Tasks**: Simple reasoning steps
- **Medium Tasks**: Multi-step reasoning
- **Hard Tasks**: Complex reasoning chains

#### Benefits for Reasoning

1. **Sample Efficiency**: Requires fewer total samples than direct learning
2. **Improved Generalization**: Better performance on unseen complex tasks
3. **Stable Training**: Reduces training instability common in RL
4. **Foundation Building**: Establishes strong reasoning fundamentals

#### Implementation Strategy

1. **Task Decomposition**: Organize reasoning tasks by difficulty
2. **Probabilistic Scheduling**: Gradually shift focus from easy to hard tasks
3. **Performance Monitoring**: Track progress across difficulty levels
4. **Adaptive Fading**: Remove easier tasks as competency develops

### Multi-Task Learning for Reasoning

#### Advantages
- **Shared Representations**: Learn common reasoning patterns across tasks
- **Transfer Learning**: Skills from one reasoning type transfer to others
- **Robustness**: Better generalization across different problem types
- **Efficiency**: Single model handles multiple reasoning tasks

#### Applications for Logical Reasoning
- **Cross-Domain Transfer**: Truth-teller problems → Seating arrangements
- **Skill Composition**: Combine logical operators across different contexts
- **Pattern Recognition**: Identify common reasoning structures

### Mixture of Experts (MoE) for Reasoning

#### Architecture Benefits
- **Specialized Experts**: Different experts for different reasoning types
- **Scalability**: Add experts without increasing inference cost proportionally
- **Efficiency**: Only activate relevant experts for specific tasks
- **Modularity**: Easy to add new reasoning capabilities

#### Expert Specialization
- **Logic Expert**: Truth-teller/liar problems, boolean reasoning
- **Spatial Expert**: Seating arrangements, spatial relationships
- **Temporal Expert**: Sequence reasoning, time-based logic
- **Relational Expert**: Family trees, relationship inference

#### Implementation Considerations
- **Router Training**: Learn to select appropriate experts
- **Load Balancing**: Ensure even expert utilization
- **Expert Diversity**: Maintain distinct specializations
- **Inference Optimization**: Minimize expert switching overhead

### Knowledge Distillation for Reasoning

#### Teacher-Student Framework
- **Large Teacher Model**: High-capacity reasoning model (e.g., GPT-4, DeepSeek-R1)
- **Small Student Model**: Target deployment model (e.g., Qwen3-4B)
- **Reasoning Transfer**: Distill step-by-step reasoning capabilities

#### Distillation Strategies

1. **Response Distillation**: Learn from final answers
2. **Process Distillation**: Learn from reasoning traces
3. **Attention Distillation**: Transfer attention patterns
4. **Feature Distillation**: Match intermediate representations

#### Benefits for 4B Models
- **Efficiency**: Maintain reasoning quality with smaller model
- **Cost Reduction**: Lower inference costs
- **Deployment**: Easier deployment in resource-constrained environments
- **Specialization**: Focus on specific reasoning domains

### Progressive Training Strategies

#### Incremental Complexity
1. **Model Size Progression**: Start with smaller models, gradually expand
2. **Context Length Progression**: Increase reasoning chain length gradually
3. **Concept Progression**: Introduce logical concepts incrementally
4. **Domain Progression**: Move from simple to complex reasoning domains

#### Implementation Framework
- **Staged Training**: Multiple training phases with increasing difficulty
- **Checkpoint Management**: Save models at each progression stage
- **Performance Validation**: Ensure competency before advancing
- **Adaptive Scheduling**: Adjust progression based on performance

### Constitutional AI for Reasoning

#### Principle-Based Training
- **Logical Consistency**: Ensure reasoning follows logical principles
- **Truthfulness**: Prioritize accurate reasoning over confident errors
- **Transparency**: Encourage explainable reasoning steps
- **Robustness**: Handle edge cases and contradictions gracefully

#### Implementation for Logical Tasks
- **Constitution Definition**: Define logical reasoning principles
- **Self-Critique**: Model evaluates its own reasoning
- **Iterative Refinement**: Improve reasoning through self-correction
- **Principle Alignment**: Ensure adherence to logical rules

### Hybrid Training Approaches

#### Combining Multiple Techniques
1. **Curriculum + GRPO**: Progressive difficulty with group optimization
2. **Multi-task + DPO**: Shared learning with preference optimization
3. **MoE + Distillation**: Expert specialization with knowledge transfer
4. **Constitutional + Curriculum**: Principle-based progressive learning

#### Synergistic Benefits
- **Complementary Strengths**: Each technique addresses different challenges
- **Robust Training**: Multiple approaches reduce single-point failures
- **Optimal Performance**: Combined techniques often outperform individual methods
- **Flexibility**: Adapt combination based on specific requirements

