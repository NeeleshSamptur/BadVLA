# Attention-based analysis: cross-attention vs. self-attention in OpenVLA-OFT

## The question

In diffusion models with text conditioning (e.g. Stable Diffusion), cross-attention is a
genuinely separate, dedicated mechanism: the image-generation network's features act as
*queries*, and a *separate* frozen text encoder's output acts as *keys/values* -- two
distinct streams, one explicitly attending to the other. That's why cross-attention maps
are such a clean interpretability tool there -- you can point at "this generated pixel
region attended most strongly to the word 'dog.'"

The question raised: can something like that be brought into this project -- and does it
even make sense given OpenVLA-OFT's architecture, since the vision features go through a
projector before reaching the LLM?

## Why OpenVLA-OFT doesn't have that structure

The projector converts vision features into the *same* embedding space as language
tokens, and then everything -- vision tokens, language tokens, the proprio token -- gets
**concatenated into one single sequence** before ever reaching the transformer. From that
point on, every layer uses plain **self-attention over the whole mixed sequence**. There
is no separate "vision stream" and "language stream" attending to each other anymore --
it's one sequence, one attention mechanism, and the distinction between "this token came
from an image patch" vs. "this token came from a word" only exists in our own bookkeeping
(which position indices correspond to which modality), not in the model's computation
itself.

So: what would be cross-attention in a dedicated-cross-attention architecture collapses
into ordinary self-attention here, once you're past the projector.

## What's still recoverable

Even without a dedicated cross-attention module, something functionally equivalent can
still be extracted, since we know exactly which sequence positions are vision-patch
tokens and which are language/action-related tokens.

Take the self-attention weight matrix at some layer, and look only at the sub-block
where the **query** is a language or action-related position and the **key** is a
vision-patch position -- that submatrix *is* the cross-modal attention pattern, just
recovered from inside a self-attention matrix rather than from a standalone module.

Concretely, this could test: when predicting the action, does attention concentrate
abnormally on the trigger patch's specific token positions, versus spreading across
task-relevant image regions normally (clean vs. triggered)?

## The real implementation catch

To get actual attention *weights* (not just the module's output), the model typically
needs to run with `attn_implementation="eager"` -- the fast, memory-efficient attention
kernels (SDPA, FlashAttention) that are almost certainly what's loaded by default do not
expose the full attention matrix at all, for efficiency reasons. Switching to eager
attention is a real change to how the model runs (likely slower, possibly needs a fresh
model load), not just a new hook.

## Status

Genuinely doable, and a different kind of signal than anything tried so far in this
project (attention allocation, not activation magnitude or vocabulary projection) -- but
it requires reloading the model in eager mode before any hook can see real attention
weights. Not yet implemented or tested. Before building the full analysis, the first
concrete step would be checking whether this checkpoint can actually be loaded with
`attn_implementation="eager"` without breaking anything else in the existing pipeline.
