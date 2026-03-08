---
layout: post
title: A Field Guide to NanoGPT Speedrun Optimizations
date: 2026-03-07
description: Improvements to the NanoGPT and NanoChat speedruns

bibliography: nanogpt-improvements.bib
toc:
  sidebar: left

_styles: >
    h3 {
        font-size: 1.5rem;
    }
---

Near the end of 2022, [Andrej Karpathy](https://karpathy.ai/) released NanoGPT[^nanogpt], a small, hackable GPT. A few years later, [Keller Jordan](https://kellerjordan.github.io/) forked off the Modded NanoGPT[^nanogptreadme] repository and launched the NanoGPT speedrun, which sought to improve NanoGPT's training speed on a single 8xH100 node as much as possible.

Since then, Modded NanoGPT has gone from taking 45 minutes to train a model equivalent to GPT-2 Small to taking less than 90 seconds. Using a similar setup, Andrej Karpathy's NanoGPT successor, NanoChat[^nanochat], has reduced the time needed to train a GPT-2 XL level model from 168 hours to around 2 hours. These improvements have massively changed the economics of small-model training, making it possible to train a fully functional model for only around $50.

This post covers some of the most interesting and impactful upgrades used in both speedruns, with the goal of showing how model training got so fast. These improvements are presented in (approximate) order of importance.

## Muon
---

Muon[^muon] is an optimizer designed for the 2D parameter matrices of neural networks, and is used in both Modded NanoGPT and NanoChat. Notably, Muon has proven to be very effective even for larger models. For example, Moonlight[^moonlight] estimates that Muon gives approximately 2x computational efficiency when compared to AdamW. For parameters that are not neural net parameter matrices, both NanoGPT and NanoChat use AdamW instead. Muon uses standard stochastic gradient descent with momentum, but orthogonalizes the update matrix using a single Newton-Schulz iteration to orthogonalize the update matrix. Why does this help so much? The original Muon blog post[^muon] conjectures that
> The updates produced by both SGD-momentum and Adam for the 2D parameters in transformer-based neural networks typically have very high condition number. That is, they are almost low-rank matrices, with the updates for all neurons being dominated by just a few directions. We speculate that orthogonalization effectively increases the scale of other “rare directions” which have small magnitude in the update but are nevertheless important for learning.

Modded NanoGPT and NanoChat use an efficient distributed version of Muon that distributes the Newton-Schulz iteration over multiple GPUs[^distributedmuon].

### Cautious Weight Decay

Cautious weight decay[^cautiousweightdecay] only applies weight decay when the update and the weight have the same sign ($\text{update} \times \text{weight} > 0$). The intuition for this is that if the signs are different, then the update is already pulling the weight back towards zero. Both Modded NanoGPT and NanoChat use cautious weight decay. One slight difference between the two is that Modded NanoGPT uses cautious weight decay for Adam as well, while NanoChat has no weight decay for Adam.

### Momentum Warmup

Both Modded NanoGPT and NanoChat gradually increase the momentum used for Muon from 0.85 to 0.95. The intuition for this is that the loss landscape changes more rapidly early on, so lower momentum is desirable[^momentumwarmup].

## Attention
---

This section covers various improvements to the attention mechanism, mainly along two lines:

1. Moving to more optimized implementations of the attention computation
2. Alterations to the size of the attention window

### FlashAttention

The FlashAttention series of optimized attention implementations gives a major speedup to attention computations[^flashattention] [^flashattention2] [^flashattention3]. Both Modded NanoGPT and NanoChat use FlashAttention 3.

### Sliding Window Attention

Both Modded NanoGPT and NanoChat use a short-long pattern for attention windows, where there are several short-window attention layers, followed by one long-window layer. Both Modded NanoGPT and NanoChat use a repeating SSSL pattern, where there are three short windows followed by a long window that is twice the length. However, the final layer is forced to always be a long window. This pattern is slightly altered for Modded NanoGPT, due to there being only 11 layers, so Modded NanoGPT only uses long-window attention on layers 4 and 11.

A similar method of alternating short and long attention windows was used in GPT-3[^gpt3].

### Attention Window Warmup

Modded NanoGPT uses an attention window schedule, where the size of the attention window is gradually increased[^windowwarmup]. One disadvantage of changing the window size when using FlashAttention 3 is that each change requires some recompilation. Due to this, Modded NanoGPT increases the window size in a few large steps throughout training.

<div class="row mt-3 justify-content-center">
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/nano-gpt/attention_window_size.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    An example of alternating attention window lengths, with window length increasing over time. Figure by <a href="https://x.com/leloykun/status/1880301758154698958">Franz Louis Cesista</a>.
</div>

## ClimbMix
---

While Modded NanoGPT does not allow changing the training data, NanoChat has experimented with several different training datasets, eventually settling on Nemotron-ClimbMix[^climbmix]. ClimbMix was created through the following procedure:
1. Begin with data from:
* Nemotron-CC[^nemotroncc], which filters Common Crawl[^commoncrawl] data for the highest-quality data
* SmolLM-Corpus[^smollm], a mixture of general educational data from FineWeb-Edu[^fineweb], programming data from The Stack[^thestack], and synthetic data
1. Cluster the data, creating 21 semantic clusters that cover a variety of different topics
1. Using a small model, learn the optimal data mixture for performance on downstream tasks

This results in a high-quality dataset that appropriately weights the frequency of various topics.

<div class="row mt-3 justify-content-center">
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/nano-gpt/topic_clusters.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Nemotron-ClimbMix topic clusters. Figure from <a href="https://arxiv.org/pdf/2504.13161#page=17">Diao et al</a>.
</div>

Modded NanoGPT uses FineWeb-Edu, a deduplicated and filtered selection of Common Crawl that was further filtered for data with high educational quality.

## Architectural Modernizations
---

This section includes architectural modifications that are already relatively well-known, and are implemented in both Modded NanoGPT and NanoChat with minimal adjustments. Additionally, many of these adjustments are fairly standard in modern open-source LLMs. In that sense, the modifications in this section can be considered to be bringing Modded NanoGPT and NanoChat up to speed with modern Transformer implementations.

### Pre-Norm

Pre-norm[^prenorm] moves the normalization blocks out of the residual stream, placing them at the start of each sub-block (attention / MLP). This controls the gradient norm and increases training stability. This was also used in GPT-2[^gpt2], along with many other models.

<div class="row mt-3 justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/nano-gpt/pre_norm_post_norm.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Post-norm (left) versus pre-norm (right). Figure from <a href="https://arxiv.org/pdf/2002.04745#page=2">Xiong et al</a>.
</div>

### QKNorm

QKNorm[^qknorm] normalizes each query vector $q_i$ and key vector $k_j$ after splitting along the head dimension and applying RoPE. While the original QKNorm paper uses the $\ell_2$-norm, both NanoChat and Modded NanoGPT apply RMSNorm instead. The primary benefit of this improvement is stopping the softmaxes in the attention computation from being easily saturated by large key or query vectors, allowing for more diverse attention patterns. This also improves training by stopping the attention gradients from growing extremely small. However, one disadvantage of this approach is that it can lead to the model not being able to sufficiently "focus" on important tokens, which has led to some later efforts to tweak the attention scale[^attentionscale].

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/nano-gpt/no_qk_norm.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/nano-gpt/qk_norm.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Attention without QK-norm (above) versus attention with QK-norm (below). Figures from <a href="https://arxiv.org/pdf/2010.04245#page=3">Henry et al</a>.
</div>

### RMSNorm

RMSNorm[^rmsnorm] normalizes all input vectors $\mathbf{a}$ according to the formula

$$\overline{a}_i = \frac{a_i}{\text{RMS}(\mathbf{a})} g_i$$

Where $a_i$ is the $i$-th value of $\mathbf{a}$, $g_i$ is a learnable gain parameter, and RMS is the root mean square of the relevant vector. Both Modded NanoGPT and NanoChat also drop the gain parameter $g_i$, yielding

$$\overline{a}_i = \frac{a_i}{\text{RMS}(\mathbf{a})}$$

RMSNorm is a simpler and faster normalization method than LayerNorm[^layernorm], which was used in the original Transformer.

### RoPE

Rotary Position Embeddings[^rope] adds positional information to the query and key vectors before performing the attention computation, rather than adding positional information when constructing the initial hidden state. More specifically, RoPE rotates pairs of dimensions in the query and key vectors based on each token's position and the rotation speed for that dimension pair. RoPE helps models learn to include positional information without adding extra parameters.

### ReLU²

The ReLU² activation function[^relu2] has been previously shown to strike a good balance between having high sparsity and good performance[^relu2wins]. ReLU² is defined as

$$\text{ReLU}^2(x) = \max(0, x)^2$$

Many modern LLMs instead use SwiGLU[^swiglu] as their activation function. SwiGLU was tested in NanoChat on several scales, but consistently gave decreased performance.

### Untied Embeddings

The original NanoGPT had tied embeddings, where the LM head matrix is the transpose of the input embedding matrix (as recommended by[^tiedembeddings]). Both Modded NanoGPT and NanoChat untied these matrices, which is common in modern LLMs. One of the reasons why untying is likely to be beneficial in this case is because it increases the parameter count without causing a corresponding increase in the number of FLOPs per pass.

As a side note, tied embeddings can cause some unexpected issues (see Neel Nanda on the SolidGoldMagikarp token for one interesting example[^solidgoldmagikarp]).

## Skip Connections & Value Embeddings
---

### Value Embeddings

Value residual learning[^valueresiduallearning] proposes modifying the computed value matrix $V_n$ on layer $n$ according to the formula

$$V'_n = \lambda_{n,1} \cdot V_n + \lambda_{n,2} \cdot V_1$$

This is essentially mixing together the values on layer $n$ with the values on layer 1, allowing access to the computed values from lower layers. A variant of this was originally implemented in Modded NanoGPT as

$$V'_n = (1 - \lambda_{n}) \cdot V_n + \lambda_{n} \cdot V_1$$

where $\lambda_{n}$ is a learnable parameter for each layer[^valueresiduallearningvariant].

Later, this formula was changed to mix $V_n$ and $VE_n$, where $VE_n$ are learnable *value embeddings* for the given layer[^valueembed]. More specifically, each layer is given an embedding matrix, which is then used to get layerwise value embeddings for each token, $VE_n$. The primary motivation for this change was to provide token-specific features to the attention values. This change replaces the original value residual learning formula with

$$V'_n = (1 - \lambda_{n}) \cdot V_n + \lambda_{n} \cdot VE_n$$

Both Modded NanoGPT and NanoChat use value embeddings. However, they structure them slightly differently:

1. NanoChat adds value embeddings to every other layer
2. Modded NanoGPT shares value embeddings in a U-Net style pattern[^unet] [^valueembedunet]. For example, in the 12 layer architecture used at the time, layers 1 and 12 would share the same value embeddings, layers 2 and 11 would have the same value embeddings, etc.

Value embeddings seem to be particularly helpful for model speedrunning because they add a large number of parameters without causing a correspondingly large increase in the number of FLOPs per token.

### Hidden State Residual Connections

Modded NanoGPT also adds skip connections for the hidden state[^valueresiduallearningvariant], updating the hidden state at the start of layer $n$ according to

$$X'_n = \lambda_{n,1} \cdot X_n + \lambda_{n,2} \cdot X_1$$

Since $X_1$ is the hidden state at the start of the first layer, this is equivalent to mixing the current layer's hidden state and the relevant token embeddings.

This was later modified to use a U-Net style architecture[^unet], adding connections to previous hidden states[^unetconnections]. This changed the hidden state residual formula to

$$X'_n = \lambda_{n,1} \cdot X_n + \lambda_{n,2} \cdot X_1 + \lambda_{n,3} \cdot X_k$$

This change was only made for $n > \frac{\text{layers}}{2}$, with $k = \text{layers} - n + 1$. For example, with the 12 layer architecture used at the time, only layers 7 through 12 would use this updated formula. Layer 6 would feed into 7, 5 into 8, and so on, up to 1 into 12.

## Logit Soft-Capping
---

Gemma 2[^gemma2] applies a soft cap to logits, using the following formula

$$\text{logits} \gets \text{soft_cap} \cdot \tanh\!\left(\frac{\text{logits}}{\text{soft_cap}}\right)$$

This limits the range of the logits to a range of $[-\text{soft\_cap}, +\text{soft\_cap}]$. One intuition for why this is helpful is that it limits the model's confidence; for small models, it is unlikely that they will be completely certain of the next token, so capping the logits stops them from expressing 100% certainty. This also helps with vanishing gradient issues.

Both Modded NanoGPT and NanoChat implement this soft cap with $\text{soft\_cap} = 15$. However, they only use the soft cap for the LM head, while Gemma 2 applies it to the attention logits as well. It is likely that the optimal value of the softcap would be larger for larger models, since the ability to express increased confidence would become more useful.

## Low Precision
---

Both Modded NanoGPT and NanoChat have attempted to reduce the precision of their models, but in different ways.

Modded NanoGPT uses FP8 for the LM head only. This was also tested in NanoChat, but did not give a significant benefit.

Instead, NanoChat uses FP8 for all linear layers, which gives a speedup of approximately 17% tokens per second during training, but takes more tokens to reach the same validation loss, resulting in the speedup being smaller overall (~5% speedup). This seems to give greater benefits for larger models, as testing full FP8 on smaller models made them slower overall. Full FP8 is most effective when using tensorwise scaling, rather than rowwise scaling.

## Miscellaneous Improvements
---

### Aligning to Multiples of 64

Both Modded NanoGPT and NanoChat inherit padded embeddings from the original NanoGPT. While the original NanoGPT had a vocabulary size of 50,257, padding to the next multiple of 64 (50,304) gave a major speedup[^powersof64]. In general, it is important to make sure that matrix dimensions are multiples of a sufficiently large power of 2 (16, 32, 64, 128, etc).

### Zero-Initialized Output Layers

In appendix D.2, *Tensor Programs V* [^tensorprogramsv] suggests initializing output layers as zero, which makes the optimal hyperparameters for different model sizes match more closely. Both Modded NanoGPT and NanoChat follow this recommendation by initializing the final attention projection layer as zero and initializing the output layer of all MLPs as zero. Interestingly, this gives a speedup on Modded NanoGPT, despite *Tensor Programs V* only recommending it as a way to improve hyperparameter transfer, and simply noting that "we do not find this modification to be detrimental to performance".

## Further Reading
---

Some other writeups and resources that may be of use:

- Andrej Karpathy's post *Beating GPT-2 for \<\<$100: the nanochat journey* [^nanochatannouncement] covers many of his initial optimizations for NanoChat and has a list of what worked and what didn't work.
- Andrej Karpathy's NanoChat experiment log[^nanochatlog] contains many useful details on what kind of optimizations were tested, and what the results were. This was the primary source for most of the NanoChat optimization details.
- The Modded NanoGPT README (primarily maintained by Keller Jordan and Larry Dial)[^nanogptreadme] covers the various speedrun world records and their associated improvements.
- Larry Dial's writeup *How the NanoGPT Speedrun WR dropped by 20% in 3 months* [^speedrunlatest] covers many of the more recent modifications to Modded NanoGPT (roughly covering the July–October range).
- Varun Srivastava's writeup *Muon in Modded NanoGPT* [^muonimprovementwriteup] covers many of the improvements made to Muon for the Modded NanoGPT speedrun.

If you found this post useful, please cite it as:
```bibtex
@article{conway2026,
  title   = {A Field Guide to NanoGPT Speedrun Optimizations},
  author  = {Evan Conway},
  year    = {2026},
  month   = {Mar},
  url     = {https://evanjayconway.com/posts/2026/nanogpt-improvements/}
}
```

## References
---

[^layernorm]: Ba et al 2016, [*Layer Normalization*](https://arxiv.org/abs/1607.06450)

[^nemotroncc]: Su et al 2024, [*Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset*](https://arxiv.org/abs/2412.02595)

[^commoncrawl]: [*Common Crawl*](https://commoncrawl.org/)

[^fineweb]: Penedo et al 2024, [*The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*](https://arxiv.org/abs/2406.17557)

[^smollm]: Allal et al 2024, [*SmolLM-Corpus*](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)

[^thestack]: Kocetkov et al 2022, [*The Stack: 3 TB of permissively licensed source code*](https://arxiv.org/abs/2211.15533)

[^climbmix]: Diao et al 2025, [*Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training*](https://arxiv.org/abs/2504.13161)

[^relu2wins]: Zhang et al 2024, [*ReLU² Wins: Discovering Efficient Activation Functions for Sparse LLMs*](https://arxiv.org/abs/2402.03804)

[^relu2]: So et al 2022, [*Primer: Searching for Efficient Transformers for Language Modeling*](https://arxiv.org/abs/2109.08668)

[^swiglu]: Shazeer 2020, [*GLU Variants Improve Transformer*](https://arxiv.org/abs/2002.05202)

[^prenorm]: Xiong et al 2020, [*On Layer Normalization in the Transformer Architecture*](https://arxiv.org/abs/2002.04745)

[^gpt2]: Radford et al 2019, [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

[^rmsnorm]: Zhang & Sennrich 2019, [*Root Mean Square Layer Normalization*](https://arxiv.org/abs/1910.07467)

[^tensorprogramsv]: Yang et al 2022, [*Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer*](https://arxiv.org/abs/2203.03466)

[^gemma2]: Gemma Team 2024, [*Gemma 2: Improving Open Language Models at a Practical Size*](https://arxiv.org/abs/2408.00118)

[^solidgoldmagikarp]: Neel Nanda 2023, [*comment on "SolidGoldMagikarp (plus, prompt generation)"*](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg?commentId=zCzd4cAkwnRBdtcYm)

[^flashattention]: Dao et al 2022, [*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*](https://arxiv.org/abs/2205.14135)

[^flashattention2]: Dao 2023, [*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*](https://arxiv.org/abs/2307.08691)

[^flashattention3]: Shah et al 2024, [*FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*](https://arxiv.org/abs/2407.08608)

[^gpt3]: Brown et al 2020, [*Language Models are Few-Shot Learners*](https://arxiv.org/abs/2005.14165)

[^muonimprovementwriteup]: Srivastava 2025, [*Muon in Modded NanoGPT*](https://varunneal.github.io/essays/muon)

[^speedrunlatest]: Dial 2025, [*How the NanoGPT Speedrun WR dropped by 20% in 3 months*](https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg/how-the-nanogpt-speedrun-wr-dropped-by-20-in-3-months)

[^nanogpt]: Karpathy 2022, [*nanoGPT*](https://github.com/karpathy/nanoGPT)

[^nanochat]: Karpathy 2025, [*nanochat: The best ChatGPT that $100 can buy*](https://github.com/karpathy/nanochat)

[^nanochatannouncement]: Karpathy 2026, [*Beating GPT-2 for <<$100: the nanochat journey*](https://github.com/karpathy/nanochat/discussions/481)

[^nanochatlog]: Karpathy 2025, [*dev/LOG.md*](https://github.com/karpathy/nanochat/blob/master/dev/LOG.md)

[^nanogptreadme]: Jordan et al 2024, [*modded-nanogpt README*](https://github.com/KellerJordan/modded-nanogpt/blob/master/README.md)

[^powersof64]: [*Karpathy post on X*](https://x.com/karpathy/status/1621578354024677377)

[^qknorm]: Henry et al 2020, [*Query-Key Normalization for Transformers*](https://arxiv.org/abs/2010.04245)

[^tiedembeddings]: Press & Wolf 2017, [*Using the Output Embedding to Improve Language Models*](https://arxiv.org/abs/1608.05859)

[^rope]: Su et al 2023, [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864)

[^windowwarmup]: [*Fern post on X*](https://x.com/hi_tysam/status/1860851011797053450)

[^muon]: Jordan et al 2024, [*Muon: An optimizer for hidden layers in neural networks*](https://kellerjordan.github.io/posts/muon/)

[^moonlight]: Liu et al 2025, [*Muon is Scalable for LLM Training*](https://arxiv.org/abs/2502.16982)

[^cautiousweightdecay]: Chen et al 2025, [*Cautious Weight Decay*](https://arxiv.org/abs/2510.12402)

[^momentumwarmup]: [*Keller Jordan post on X*](https://x.com/kellerjordan0/status/1854305895074947465)

[^attentionscale]: [*Franz Louis Cesista post on X*](https://x.com/leloykun/status/1880301762952958435)

[^valueresiduallearning]: Zhou et al 2025, [*Value Residual Learning*](https://arxiv.org/abs/2410.17897)

[^valueresiduallearningvariant]: [*Grad62304977 post on X*](https://x.com/Grad62304977/status/1854295837741809933)

[^unetconnections]: [*Brendan Hogan post on X*](https://x.com/brendanh0gan/status/1855273758681866352)

[^visualizinglosslandscapeneural]: Li et al 2018, [*Visualizing the Loss Landscape of Neural Nets*](https://arxiv.org/abs/1712.09913)

[^unet]: Ronneberger et al 2015, [*U-Net: Convolutional Networks for Biomedical Image Segmentation*](https://arxiv.org/abs/1505.04597)

[^valueembed]: [*Braden Koszarsky post on X*](https://x.com/KoszarskyB/status/1864746625572257852)

[^valueembedreason]: [*Braden Koszarsky post on X*](https://x.com/KoszarskyB/status/1864760973078266262)

[^valueembedunet]: [*You Jiacheng post on X*](https://x.com/YouJiacheng/status/1865761473886347747)

[^distributedmuon]: [*Keller Jordan post on X*](https://x.com/kellerjordan0/status/1847291684016783746)