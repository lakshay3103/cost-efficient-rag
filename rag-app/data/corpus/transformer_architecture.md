# Transformer (Deep Learning Architecture)

The Transformer is a deep learning architecture that relies primarily on the attention mechanism to draw global dependencies between input and output.

## Origin

The Transformer architecture was introduced in the paper "Attention Is All You Need," published in 2017 by researchers at Google. The authors included Ashish Vaswani, Noam Shazeer, Niki Parmar, and several others. Before Transformers, sequence modeling tasks such as machine translation were dominated by recurrent neural networks (RNNs) and long short-term memory (LSTM) networks, which process sequences token by token and are difficult to parallelize during training.

## Key Innovation: Self-Attention

The core innovation of the Transformer is the self-attention mechanism, which allows the model to weigh the importance of different tokens in a sequence relative to each other, regardless of their distance in the sequence. This removes the sequential processing bottleneck of RNNs and allows for much greater parallelization during training on modern hardware such as GPUs.

## Architecture Components

A standard Transformer consists of an encoder and a decoder, each composed of a stack of identical layers. Each encoder layer contains a multi-head self-attention sub-layer and a position-wise feed-forward network sub-layer, with residual connections and layer normalization applied around each. Because self-attention has no inherent sense of token order, Transformers add positional encodings to the input embeddings to inject information about the position of tokens in the sequence.

## Multi-Head Attention

Multi-head attention runs several attention mechanisms in parallel ("heads"), each learning to focus on different types of relationships between tokens. The outputs of the heads are concatenated and linearly transformed to produce the final output of the multi-head attention sub-layer.

## Impact and Successor Models

The Transformer architecture became the foundation for a wide range of influential models. BERT (Bidirectional Encoder Representations from Transformers), released by Google in 2018, uses only the encoder stack and is trained using masked language modeling. The GPT (Generative Pre-trained Transformer) family of models, developed by OpenAI, uses only the decoder stack and is trained to predict the next token in a sequence. GPT-3, released in 2020, had 175 billion parameters and demonstrated strong few-shot learning capabilities.

## Computational Cost

A well-known limitation of the original Transformer architecture is that the computational and memory cost of self-attention scales quadratically with the length of the input sequence, since every token must attend to every other token. This has motivated a range of research into more efficient attention mechanisms, such as sparse attention and linear attention approximations, to handle longer input sequences more efficiently.
