# Memory Network

Paper: https://arxiv.org/abs/1410.3916



## Motivation

Most machine learning models lack **an easy way to read and write to part of a (potentially very large) long-term memory component**, and to combine this seamlessly with inference. 

The central idea is to combine the successful learning strategies developed in the machine learning literature for inference with a memory component that can be read and written to. The model is then trained to learn how to operate effectively with the memory component. 

​	

## Components

A memory network consists of a memory **m** (an array of objects indexed by $m_i$) and four (potentially learned) components I, G, O and R as follows:

* I: (input feature map) – converts the incoming input to the internal feature representation. It can make use of standard pre-processing, e.g., parsing, coreference and entity resolution for text inputs, embedding.
* G: (generalization) – updates old memories given the new input. There is an opportunity for the network to compress and generalize its memories at this stage for some intended future use. The simplest form of G is to store I(x) in a “slot” in the memory: $\mathbf{m}_{H(x)}=I(x)$. More sophisticated variants of G could go back and update earlier stored memories
  * If the memory is huge, one needs to organize the memories: they can operate on only a retrieved subset of candidates.
  * If the memory becomes full, a procedure for “forgetting” could also be implemented by H as it chooses which memory is replaced.
* O: (output feature map) – produces a new output (in the feature representation space), given the new input and the current memory state.
* R: (response) – converts the output into the response format desired.



the flow of the model:

1. Convert x to an internal feature representation I(x).
2. Update memories $\mathbf{m}_{i}$ given the new input: $\mathbf{m}_{i}=G\left(\mathbf{m}_{i}, I(x), \mathbf{m}\right), \forall i$
3. Compute output features $o$ given the new input and the memory: $o=O(I(x), \mathbf{m})$
4. Finally, decode output features $o$ to give the final response: $r=R(o)$



