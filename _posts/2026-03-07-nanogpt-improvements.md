---
layout: post
title: NanoGPT Speedrun Improvements
date: 2026-03-07
description: Improvements to the NanoGPT and NanoChat speedruns

bibliography: nanogpt-improvements.bib
toc: true
---

## Fast Training for Dummies

This writeup discusses a lot of different optimizations. But which ones give the largest impact? In approximate order of impact:

1. Switch to using the Muon optimizer with the improvements discussed in the Muon section
2. Apply all the modifications in the Architectural Modernizations section
3. Update PyTorch to the latest version and use the latest FlashAttention release for your attention computation
4. Check all your tensors to make sure that their dimensions are multiples of a sufficiently large power of two
5. Tune your learning rate
6. Apply a soft cap to your logits, as described in the Logit Soft-Capping section
7. Shift your model's linear layers to use FP8, as discussed in the Low Precision section

## Architectural Modernizations

This section includes architectural modifications that are already relatively well-known, and are implemented in both Modded NanoGPT and NanoChat with minimal adjustments. Additionally, many of these adjustments are fairly standard in modern open-source LLMs. In that sense, the modifications in this section can be considered to be bringing Modded NanoGPT and NanoChat up to speed with modern Transformer implementations.

### Pre-Norm

Pre-norm <d-cite key="prenorm"></d-cite> moves the normalization blocks out of the residual stream, placing them at the start of each sub-block (attention / MLP). This controls the gradient norm and increases training stability. This was also used in GPT-2 <d-cite key="gpt2"></d-cite>, along with many other models.

**TODO: add figure showing the move**

### QKNorm

QKNorm <d-cite key="qknorm"></d-cite> normalizes each query vector $q_i$ and key vector $k_j$ after splitting along the head dimension and applying RoPE. While the original QKNorm paper uses the $\ell_2$-norm, both NanoChat and Modded NanoGPT apply RMSNorm instead. The primary benefit of this improvement is stopping the softmaxes in the attention computation from being easily saturated by large key or query vectors. This also improves training by stopping the attention gradients from growing extremely small. However, one disadvantage of this approach is that it can lead to the model not being able to sufficiently "focus" on important tokens <d-cite key="attentionscale"></d-cite>.

**TODO: add Figures 1 and 2 from QK-Norm paper**

### RMSNorm

RMSNorm <d-cite key="rmsnorm"></d-cite> normalizes all input vectors $\mathbf{a}$ according to the formula

$$\overline{a}_i = \frac{a_i}{\text{RMS}(\mathbf{a})} g_i$$

Where $a_i$ is the $i$-th value of $\mathbf{a}$, $g_i$ is a learnable gain parameter, and RMS is the root mean square of the relevant vector. Both Modded NanoGPT and NIanoChat also drop the gain parameter $g_i$, yielding

$$\overline{a}_i = \frac{a_i}{\text{RMS}(\mathbf{a})}$$

### RoPE

Rotary Position Embedding <d-cite key="rope"></d-cite> adds positional information to the query and key vectors before performing the attention computation. More specifically, RoPE rotates pairs of dimensions in the query and key vectors based on that token's position and the rotation speed for that dimension pair.

**TODO: feel like I should add more here**

### ReLU²

The ReLU² activation function <d-cite key="relu2"></d-cite> has been previously shown to strike a good balance between having high sparsity and good performance <d-cite key="relu2wins"></d-cite>. ReLU² is defined as

$$\text{ReLU}^2(x) = \max(0, x)^2$$

Many modern LLMs instead use SwiGLU <d-cite key="swiglu"></d-cite> as their activation function. This was tested in NanoChat on several scales, but consistently gave decreased performance.

### Untied Embeddings

The original NanoGPT had tied embeddings, where the LM head matrix is the transpose of the input embedding matrix (as recommended by <d-cite key="tiedembeddings"></d-cite>). Both Modded NanoGPT and NanoChat untied these matrices. One of the reasons why untying is likely to be beneficial in this case is because it increases the parameter count without causing a corresponding increase in the number of FLOPs per pass.

As a side note, tied embeddings can cause some unexpected issues (see Neel Nanda on the SolidGoldMagikarp token for one interesting example <d-cite key="solidgoldmagikarp"></d-cite>).

## Attention

This section covers various improvements to the attention mechanism, mainly along two lines:

1. Moving to more optimized implementations of the attention computation
2. Alterations to the size of the attention window

### FlashAttention

The FlashAttention series (<d-cite key="flashattention"></d-cite>, <d-cite key="flashattention2"></d-cite>, <d-cite key="flashattention3"></d-cite>) of optimized attention implementations gives a major speedup to attention computations. Both Modded NanoGPT and NanoChat use the latest version, FlashAttention 3.

**TODO: update to FlashAttention 4?**

### Sliding Window Attention

Both Modded NanoGPT and NanoChat use a short-long pattern for attention windows, where there are several short-window attention layers, followed by one long-window layer. Both Modded NanoGPT and NanoChat use a repeating SSSL pattern, where there are three short windows followed by a long window that is twice the length. However, the final layer is forced to always be a long window. This pattern is slightly altered for Modded NanoGPT, due to there being only 11 layers, so Modded NanoGPT only uses long-window attention on layers 4 and 11.

A similar method of alternating short and long attention windows was used in GPT-3 <d-cite key="gpt3"></d-cite>.

### Attention Window Warmup

Modded NanoGPT uses an attention window schedule, where the size of the attention window is gradually increased. This was originally proposed by Fern <d-cite key="windowwarmup"></d-cite>. One disadvantage of changing the window size when using FlashAttention 3 is that each change requires some recompilation. Due to this, Modded NanoGPT increases the window size in a few large steps throughout training.

## Logit Soft-Capping

Gemma 2 <d-cite key="gemma2"></d-cite> applies a soft cap to logits, using the following formula

$$\text{logits} \gets \text{soft\_cap} \cdot \tanh\!\left(\frac{\text{logits}}{\text{soft\_cap}}\right)$$

This limits the range of the logits to a range of $[-\text{soft\_cap}, +\text{soft\_cap}]$. Both Modded NanoGPT and NanoChat implement this soft cap with $\text{soft\_cap} = 15$. However, they only use the soft cap for the LM head, while Gemma 2 applies it to the attention logits as well. It is likely that the optimal value of the softcap would be larger for larger models, since the ability to express increased confidence would become more useful.

## Low Precision

This section covers efforts to reduce the numerical precision of various parts of Modded NanoGPT / NanoChat.

### FP8 Head

Modded NanoGPT uses FP8 for the LM head only. This was also tested in NanoChat, but did not give a significant benefit. One interesting observation from the NanoChat testing was that GPU memory increased by approximately 2GB for unknown reasons.

### Full FP8

NanoChat uses FP8 for all linear layers, which gives a speedup of approximately 17% tokens per second during training, but takes more tokens to reach the same validation loss, resulting in the speedup being smaller overall (~5% speedup). This seems to give greater benefits for larger models, as testing full FP8 on smaller models made them slower overall. Full FP8 is most effective when using tensorwise scaling, rather than rowwise scaling.

## Muon

Muon <d-cite key="muon"></d-cite> is an optimizer designed for the 2D parameter matrices of neural networks, and is used in both Modded NanoGPT and NanoChat. Notably, Muon has proven to be very effective even for larger models. For example, Moonlight <d-cite key="moonlight"></d-cite> estimates that Muon gives approximately 2x computational efficiency when compared to AdamW. For parameters that are not neural net parameter matrices, both NanoGPT and NanoChat use AdamW instead. Muon uses standard stochastic gradient descent with momentum, but orthogonalizes the update matrix using a Newton-Schulz iteration.

Modded NanoGPT and NanoChat also use an efficient distributed version of Muon that distributes the Newton-Schulz iteration over multiple GPUs <d-cite key="distributedmuon"></d-cite>.

### Cautious Weight Decay

Cautious weight decay <d-cite key="cautiousweightdecay"></d-cite> only applies weight decay when the update and the weight have the same sign ($\text{update} \times \text{weight} > 0$). The intuition for this is that if the signs are different, then the update is already pulling the weight back towards zero. Both Modded NanoGPT and NanoChat use cautious weight decay. One slight difference between the two is that Modded NanoGPT uses cautious weight decay for Adam as well, while NanoChat has no weight decay for Adam.

### Momentum Warmup

Both Modded NanoGPT and NanoChat gradually increase the momentum used for Muon from 0.85 to 0.95. The intuition for this is that the loss landscape changes more rapidly early on, so lower momentum is desirable <d-cite key="momentumwarmup"></d-cite>.

## Skip Connections

### Value Embeddings

Value residual learning <d-cite key="valueresiduallearning"></d-cite> proposes modifying the computed value matrix $V_n$ on layer $n$ according to the formula

$$V'_n = \lambda_{n,1} \cdot V_n + \lambda_{n,2} \cdot V_1$$

This is essentially mixing together the values on layer $n$ with the values on layer 1, allowing access to the computed values from lower layers. A variant of this was originally implemented in Modded NanoGPT as

$$V'_n = (1 - \lambda_{n}) \cdot V_n + \lambda_{n} \cdot V_1$$

where $\lambda_{n}$ is a learnable parameter for each layer <d-cite key="valueresiduallearningvariant"></d-cite>.

Later, this formula was changed to mix $V_n$ and $VE_n$, where $VE_n$ are learnable *value embeddings* for the given layer <d-cite key="valueembed"></d-cite>. More specifically, each layer is given an embedding matrix, which is then used to get layerwise value embeddings for each token, $VE_n$. The primary motivation for this change was to provide token-specific features to the attention values. This change replaces the original value residual learning formula with

$$V'_n = (1 - \lambda_{n}) \cdot V_n + \lambda_{n} \cdot VE_n$$

Both Modded NanoGPT and NanoChat use value embeddings. However, they structure them slightly differently:

1. NanoChat adds value embeddings to every other layer
2. Modded NanoGPT shares value embeddings in a U-Net style pattern <d-cite key="unet"></d-cite> <d-cite key="valueembedunet"></d-cite>. For example, in the 12 layer architecture used at the time, layers 1 and 12 would share the same value embeddings, layers 2 and 11 would have the same value embeddings, etc.

Value embeddings seem to be particularly helpful for model speedrunning because they add a large number of parameters without causing a correspondingly large increase in the number of FLOPs per token.

### Embedding Residual Connections

Modded NanoGPT also adds skip connections for the hidden state <d-cite key="valueresiduallearningvariant"></d-cite>, updating the hidden state at the start of layer $n$ according to

$$X'_n = \lambda_{n,1} \cdot X_n + \lambda_{n,2} \cdot X_1$$

Since $X_1$ is the hidden state at the start of the first layer, this is equivalent to mixing the current layer's hidden state and the relevant token embeddings.

This was later modified to use a U-Net style architecture <d-cite key="unet"></d-cite>, adding connections to previous hidden states <d-cite key="unetconnections"></d-cite>. This changed the hidden state residual formula to

$$X'_n = \lambda_{n,1} \cdot X_n + \lambda_{n,2} \cdot X_1 + \lambda_{n,3} \cdot X_k$$

This change was only made for $n > \frac{\text{layers}}{2}$, with $k = \text{layers} - n + 1$. For example, with the 12 layer architecture used at the time, only layers 7 through 12 would use this updated formula. Layer 6 would feed into 7, 5 into 8, and so on, up to 1 into 12.

## Miscellaneous Improvements

### Aligning to Multiples of 64

Both Modded NanoGPT and NanoChat inherit padded embeddings from the original NanoGPT. While the original NanoGPT had a vocabulary size of 50,257, padding to the next multiple of 64 (50,304) gave a major speedup <d-cite key="powersof64"></d-cite>. In general, it is important to make sure that matrix dimensions are multiples of a sufficiently large power of 2 (16, 32, 64, 128, etc).

### Zero-Initialized Output Layers

In appendix D.2, *Tensor Programs V* <d-cite key="tensorprogramsv"></d-cite> suggests initializing output layers as zero, which makes the optimal hyperparameters for different model sizes match more closely. Both Modded NanoGPT and NanoChat follow this recommendation by initializing the final attention projection layer as zero and initializing the output layer of all MLPs as zero. Interestingly, this gives a speedup on Modded NanoGPT, despite *Tensor Programs V* only recommending it as a way to improve hyperparameter transfer, and simply noting that "we do not find this modification to be detrimental to performance".

## Further Reading

Some other writeups and resources that may be of use:

- Andrej Karpathy's post *Beating GPT-2 for \<\<$100: the nanochat journey* <d-cite key="nanochatannouncement"></d-cite> covers many of his initial optimizations for NanoChat and has a list of what worked and what didn't work.
- Andrej Karpathy's NanoChat experiment log <d-cite key="nanochatlog"></d-cite> contains many useful details on what kind of optimizations were tested, and what the results were. This was the primary source for most of the NanoChat optimization details.
- The Modded NanoGPT README (primarily maintained by Keller Jordan and Larry Dial) <d-cite key="nanogptreadme"></d-cite> covers the various speedrun world records and their associated improvements.
- Larry Dial's writeup *How the NanoGPT Speedrun WR dropped by 20% in 3 months* <d-cite key="speedrunlatest"></d-cite> covers many of the more recent modifications to Modded NanoGPT (roughly covering the July–October range).
- Varun Srivastava's writeup *Muon in Modded NanoGPT* <d-cite key="muonimprovementwriteup"></d-cite> covers many of the improvements made to Muon for the Modded NanoGPT speedrun.