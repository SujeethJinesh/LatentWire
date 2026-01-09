# Telepathy Project: Word-for-Word Talking Points

## 🎯 One-Sentence Summary
**"I built a neural bridge that lets different LLMs communicate directly through learned soft tokens instead of text, achieving 22× speedup with higher accuracy on classification tasks."**

---

## 📊 The Core Problem (30 seconds)

**What to say:**
"When LLMs need to communicate today - like in multi-agent systems - they generate text token by token, which the other model then has to re-tokenize and process. For a simple classification task, this takes 835 milliseconds. It's like two people who speak different languages having to write everything down and translate it word by word."

---

## 🔬 What I Built (1 minute)

**What to say:**
"I built Telepathy - a learned neural bridge between heterogeneous LLMs. Specifically, I take hidden states from Llama 3.1 8B at layer 31, compress them through a Perceiver Resampler down to just 8 soft tokens, and feed those directly into Mistral 7B as input embeddings.

The Perceiver uses cross-attention where 8 learned query vectors extract information from the source model's hidden states. These 8 tokens are all Mistral needs to understand what Llama is trying to communicate.

The entire bridge is only 350 million parameters - about 2.3% of the base model size - and takes 37 milliseconds for the same classification task that used to take 835 milliseconds through text."

---

## 📈 Key Results (1 minute)

**What to say:**
"On multi-class classification tasks, we achieved remarkable results:
- AG News 4-class topic classification: 89.5% accuracy, beating prompt tuning by 7 percentage points
- TREC 6-class question classification: 96% accuracy with super-additive performance - the bridge actually outperformed both models individually by 28 percentage points
- The system is 22 times faster than text-based communication
- We achieve 4-8x compression while improving accuracy

But here's the interesting part - we discovered inverse scaling. Using fewer soft tokens actually works better. 8 tokens outperform 32 tokens, which outperform 128 tokens. The information bottleneck acts as beneficial regularization."

---

## 🏗️ Technical Architecture Details (if asked)

**What to say:**
"The architecture has three main components:

First, a statistical normalizer that handles the magnitude mismatch - Llama's hidden states range from plus-minus 20, while Mistral expects plus-minus 100.

Second, the Perceiver Resampler with 2 layers of cross-attention followed by self-attention. Each layer has 8 attention heads operating on 4096-dimensional vectors.

Third, output scaling to match Mistral's expected embedding statistics - we use RMS normalization with a learned scale parameter.

We tried 15 different architectures including diffusion transformers, vector quantization, and various auxiliary losses. The simple Perceiver approach worked best."

---

## 💡 Key Discoveries (45 seconds)

**What to say:**
"We made three surprising discoveries:

First, cross-model communication actually works better than same-model. The heterogeneity provides beneficial regularization.

Second, classification tasks work beautifully at 96% accuracy, but reasoning tasks fail completely at 2%. The bottleneck can preserve discriminative information but not sequential reasoning chains.

Third, we solved the 'magnitude shock' problem where models have 5x different scales, the vocabulary density mismatch where one model uses 128,000 tokens and the other uses 32,000, and the position encoding incompatibility between different RoPE configurations."

---

## ⚠️ Honest Limitations (30 seconds)

**What to say:**
"This doesn't work for everything. Binary classification fails - we get 49.5% on sentiment analysis, basically random. Reasoning tasks fail catastrophically - 2% on grade-school math. And it requires task-specific training - there's no zero-shot transfer yet.

The fundamental limitation is that 8 tokens can't preserve multi-step reasoning chains. But for classification and structured tasks, it's remarkably effective."

---

## 🔄 How This Relates to Hardware/FSM Work

**What to say:**
"FSM state transitions are fundamentally classification problems - given current state and input, classify the next state. That's exactly where our bridge excels.

If you're using multiple models in your RTL generation pipeline - like GPT for specifications and Claude for implementation - they could communicate directly through learned bridges instead of text. This could make your verification loops 22 times faster.

Also, our information bottleneck discovery suggests that constraining the representation might actually help generate valid RTL by preventing the model from exploring invalid state spaces."

---

## 🚀 What I Can Bring to Your Project

**What to say:**
"I can contribute three things:

First, expertise in cross-model communication. If you're using multiple models, I can make them talk directly.

Second, experience with large-scale training infrastructure. I know how to run thousands of experiments efficiently and handle multi-GPU training.

Third, a deep understanding of attention mechanisms and how to debug them. When your model isn't learning, I can figure out if it's gradient flow, attention saturation, or distribution mismatch."

---

## 💬 Conversation Starters

**If there's a lull, say:**
- "Have you considered using different models for different parts of the RTL generation? The bridge could enable specialization."
- "How much time do you currently spend on verification loops? The 22x speedup could be game-changing there."
- "I noticed you use temperature sampling - we found that information bottlenecks actually improve validity. Have you explored constraining the generation space?"

---

## 📚 If Asked About Background/Process

**What to say:**
"This was a 6-month journey through 19 experimental phases. We started trying to build a universal text compressor - that failed completely. Then we pivoted to cross-model translation for math problems - that worked but only partially.

The breakthrough came when we focused on classification tasks and discovered that less is more - 8 tokens beat 128 tokens consistently. We tested on over 50,000 examples across multiple datasets to validate this.

The code is open source, we're submitting to MLSys 2025, and the entire system runs on a single H100 GPU."

---

## 🎤 Elevator Pitch (15 seconds)

**If you only have a moment:**
"I made Llama and Mistral talk to each other directly through 8 learned tokens instead of hundreds of text tokens. It's 22 times faster and more accurate. This could revolutionize multi-model pipelines like yours."

---

## 📝 Numbers to Remember

- **22×** faster (37ms vs 835ms)
- **96%** accuracy on TREC-6
- **8** soft tokens optimal
- **2.3%** parameter overhead (350M params)
- **15** architectures tried
- **6** months of research
- **50,000+** examples tested

---

## 🎯 Closing Statement

**End with:**
"I'm excited about the intersection of neural communication and hardware synthesis. Your FSM work is dealing with structured transformations, which is exactly where our approach shines. I'd love to explore how direct model communication could accelerate your RTL generation pipeline."