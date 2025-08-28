
# Satire on Demand: Fine-Tuning a Small LLM for Satirical News Generation

[![Hugging Face Models](https://img.shields.io/badge/Hugging%20Face-Models-yellow)](https://huggingface.co/YOUR-USERNAME) <!--- Replace with your Hugging Face profile link --->
[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/YOUR-USERNAME/YOUR-SPACE) <!--- Optional: Link to a Gradio/Streamlit demo --->

This document details the process of fine-tuning the [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) model to generate high-quality satirical news headlines in the style of *The Onion*. The final model demonstrates a significant improvement over the baseline, producing satire that is clever, nuanced, and context-aware.

## The Baseline's Limitations

The base `Qwen3-4B-Instruct-2507` model, while a powerful instruction-follower, struggles with generating satirical headlines. When prompted, its outputs are formulaic, defaulting to standard tropes and relying heavily on pure absurdity.

**Baseline Zero-Shot Examples:**
```
> "Government Announces Plan to Mandate Sunglasses for All Citizens..."
> "Scientists Discover that Humans That Are Actually Just Overgrown Houseplants..."
> "Scientists Announce They've Found the World's First Self-Replicating Squirrel..."
```
These headlines lack the incisive social commentary and subtle, deadpan delivery that characterize high-quality satire.

## The Development Process

This project followed an iterative path, with each model building on the lessons of the last.

### Phase 1: Initial Exploration (Model T01)

The first major goal was to set up a baseline fine-tuning pipeline. While that may seem like an easy goal, subtle errors in implementation details or hyperparameter selection can lead to significant changes in model performance. I turned to Hugging Face's excellent LLM ecosystem:

- `transformers` - For loading and running the models.
- `datasets` - For loading and transforming the dataset.
- `bitsandbytes` - To handle quantization. 
- `peft` - To create a LoRA wrapper around the model.
- `trl` - To handle the training.

For my primary training dataset, I used [`Biddls/Onion_News`](https://huggingface.co/datasets/Biddls/Onion_News), containing 33,000 articles scraped from *The Onion* in 2022.

I needed to resolve several subtle issues to achieve a correct finetuning pipeline:

- The result of my first run had multiple special tokens, such as `<|think|>` and `<|tool_call|>` littered throughout the generated text. I realized that while the model is not intended to generate `<think></think>` blocks, the chat template was inserting them. I needed to create my own corrected chat template.
- In my second run, I attempted to fix the chat template and to increase the quantization of the model weights to `NF4`. The results were much less coherent. It seems like smaller models are harder to quantize. A third run at 8-bit quantization resolved the issue.

> #### Guided Generation
> I'd like to pull out this significant issue into its own section. In my first three runs, I used a fixed prompt: "Write a satirical headline in the style of *The Onion*." It had issues with following guided ("Write a headline about office work."), so I tried to vary the prompts. 
>
> I experimented with first [yake](https://github.com/LIAAD/yake) and then [SpaCy](https://spacy.io/) to extract the subject of the headline and insert it into the prompt ("Write a satirical headline about {SUBJECT}"). However, this NLP pipeline didn't always result in sensible or grammatically correct prompt ("Write a satirical headline about FDA"). The results were ironically worse than the fixed prompt and likely compounded by the training on prompt bug (see Phase 2 for details).
>
> After that, I tried a much simpler approach. Instead of a fixed prompt, I used a fixed pool of 6-7 prompts ("Please write a satirical headline", "Can you write a headline that sounds like The Onion?", etc.). The results were surprisingly much better and effectively resolved the generation issue.

**Finetune Examples**
```
> "Man Who Wrote The Great Gatsby Dies At 113."
> "Biden Announces He Will Not Be Running For Re-Election."
> "New Jersey Man Arrested After Shooting Up Store For Fun."
```

While the headlines are much more varied, they often sound like standard news headlines, with minimal satirical wit or social commentary. In other words, it copied the style but not the substance.

### Phase 2: Optimization (Model T02)

The baseline Model T01 was quite slow. The first several runs took 3 hours each. This significantly increased the friction in iterating, and development much more frustrating. In one situation, a small typo in the chat template (`<im_end>`) invalidated a whole run.

My focus in this stage was optimizing the runtime. There were three significant jumps in performance.
1. Setting the batch size to `8` from the initial, conservative `2` reduced the runtime to 1 hour - 1 hour 30 minutes. Training stability was not affected because we accumulated gradients over multiple batches.
2. Using [unsloth](https://docs.unsloth.ai/). It really does feel like downloading more performance. Switching the pipeline to use unsloth reduced the runtime to 30 minutes or less.
3. Training on less data. I was previously training on all 33,000 examples. However, the training loss wasn't significantly decreasing after the first few hundred examples. I ignored it initially because I know some model can continue to learn even after the loss bottoms out. However, this was not one of those cases. By using only 800 examples, we reduced the runtime to less than a minutes, with no noticeable loss of performance.

#### Training on Prompt Bug
Switching to unsloth ended up resolving a significant bug. In my original pipeline, the finetuning process computed the loss on both the prompt and the response. However, the example provided by unsloth explicitly addressed this. At first, this turned me away from unsloth because the loss was significantly higher. However, further inquiry eventually revealed the cause.

### Phase 3: Adding Substance (Model T03)

As we saw earlier, the initial finetune captured the style of Onion headlines, it lacks the substance. My first approach was to simply use a bigger model. I replaced the 4B parameter with [`Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B), but the results of finetuning were not any better.

However, that led me to experiment with enabling thinking mode / chain of thought. I researched methods for writing satirical headlines and read multiple articles. At its core, there are 3 primary steps.

1. Identify the **subject** of satire: Who or what are we poking fun at?
2. Brainstorm a **satirical angle**: What rhetorical device can we apply to the subject (e.g. exaggeration, irony, understatement, etc.)?
3. Compose the **headline**: How can we distill the subject and satirical angle into a single snappy title.

While prompting the baseline `Qwen3-8B` with that chain of thought resulted in a few gems, it was often prone to run-on generation and there were significant issues with the model's thought process.

> #### **Synthetic Chain of Thought**
> To solve this, I implemented a **knowledge distillation** pipeline:
> 
> 1.  I evaluated multiple models, and selected `Gemini 2.5 Flash` as the teacher model.
> 2.  For each headline, I prompted `Gemini 2.5 Flash` to generate a **Chain of Thought** analysis, identifying the subject and the satirical angle. Constructing an effective prompt required multiple iterations and hand-written examples.
> 3.  I concatenated the subject, satirical angle, and headline to construct a synthetic chain of thought.
> 4.  Finally, I fine-tuned the `Qwen3-4B` student model on the synthetic chain of thought prompts, allowing it to learn not only the headline, but also the conceptual steps that led to it.
>
> Training on only 240 examples still resulted in better performance than any previous run.
>
> **Chain of Thought Finetune Examples**
> ```
> "Dog Owner Sues Neighbor For Losing Dog"
> "Bush Wasn't As Bad As People Say"
> "Art Critic Fooled By Simple Painting, Event A Major Cultural Moment"
> ```
>
> While not perfect, we can easily observe a significant improvement in substance.

## The Evaluation

A key challenge was standardizing the evaluation. I selected `Gemini 2.5 Flash` to act as judge. Initial attempts resulted in the judge preferring simple style over complex substance. In particular, the humor and absurdity of the baseline samples are far easier to understand compared to the nuance of Onion News headlines.

I needed to engineer a more sophisticated, multi-step prompt that forced the judge to analyze each headline's subject and satirical angle *before* making a preference. This "Chain-of-Thought" evaluation process proved necessary for creating a somewhat reliable assessment that aligned with my human evaluation. The final prompt can be found in `prompts/evaluation.txt` and the process in `Evaluation.ipynb`. It's quite likely that a better frontier model could significantly improve the evaluator.

## Results: A Clear Hierarchy of Quality

The final models were evaluated against the baseline and a holdout set of 50 Onion headlines (labeled `Golden`). The holdout set was collected by manually visiting *The Onion* website and copying headlines.

### Quantitative Comparison

The LLM-as-judge revealed a clear and consistent improvement at each stage of the project.

| Comparison | Winner | Preference Rate |
| :--- | :--- | :--- |
| Finetune (T02) vs. Baseline | **Finetune (T02)** | **~60%** |
| CoT Finetune (T03) vs. Finetune (T02) | **CoT Finetune (T03)** | **~55%** |
| CoT Finetune (T03) vs. The Onion | **The Onion** | **~65%** |
| The Onion vs. Baseline | **The Onion** | **~90%** |

### Qualitative Comparison

The quality gap is most apparent in head-to-head examples.

**Prompt:** *"Write a satirical headline."*
- **Baseline:**    "Local Government Announces Plan To Pave Over Sun"
- **T02:**    "Man In 2012 Already Knows He'll Be Dead By 2018"
- **T03:**    "Critic Says 'Small Painting' By Koons Is 'The Most Important Art Work Of The Decade'"

Overall, here's how I would describe each model.
-   **Baseline:** The base model consistently defaulted to formulaic absurdities (e.g., "local government overreacts").
-   **T02:** The finetuned model learned the linguistic pattern, but not the substance of *The Onion*.
-   **T03:** The chain of thought model represents a significant leap, showing that a small model can learn nuanced, high-level satire when trained on a high-quality, synthetic dataset.
-   **The Onion:** *The Onion* is still a cut above, thanks to its consistently incisive commentary on *current* events. The T03 model performs well on its training data (pre-2023). However, the model's consistency could be improved and many of the topics it selects are somewhat dated.

## Repository Structure

```
.
├─ Chain-Of-Thought.ipynb    # Notebook for generating the synthetic CoT dataset.
├─ Evaluation.ipynb          # Notebook for running the final LLM-as-judge evaluation.
├─ Inference.ipynb           # Notebook to load any model and generate headlines.
├─ Parsing.ipynb             # Notebook containing various experiments in parsing headlines.
├─ T01-baseline.ipynb        # Training script (baseline).
├─ T02-unsloth.ipynb         # Training script using unsloth.
├─ T03-CoT.ipynb             # Training script using a synthetic chain of thought prompt.
├─ lib/                      # Reusable Python code for inference and preprocessing.
├─ prompts/                  # Text files containing the prompts used for generation and evaluation.
├─ results/                  # Raw CSV outputs from the LLM-as-judge evaluations.
├─ test-data/                # Holdout data and generated samples for qualitative review.
└─ training-data/            # The synthetic CoT dataset and other cached training input.
```

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Joshua-Chin/YOUR-REPO-NAME
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Generate Headlines:**
    The easiest way to test the models is to use the `Inference.ipynb` notebook. You can specify the path to a local model or download one directly from the Hugging Face Hub.

To run `Chain-Of-Thought.ipynb` or `Evaluation.ipynb`, you need a [Google AI Studio](https://aistudio.google.com) API key. Please set it as an environment variable named `GEMINI_API_KEY`.