# <i>UMP:</i> Unified Modality-aware Prompt Tuning for Text-Video Retrieval
[Haonan Zhang](https://zchoi.github.io/), [Pengpeng Zeng](https://ppengzeng.github.io/), [Lianli Gao](https://lianligao.github.io/), [Jingkuan Song](https://cfm.uestc.edu.cn/~songjingkuan/), [Heng Tao Shen](https://cfm.uestc.edu.cn/~shenht/)

[`arXiv`](https://arxiv.org/abs/2310.08446) | [`BibTeX`](#bibliography)

This is an official PyTorch implementation of the paper **UMP: Unified Modality-aware Prompt Tuning for Text-Video Retrieval** (under review). In this work, we

- present UMP, a simple yet effective method that extends prompt tuning with pre-trained models for fast adaptation to text-video retrieval.
- devise a lightweight UPG module that generates modality-aware prompt tokens to facilitate the learning of a well-aligned unified representation.
- introduce a parameter-free STS module that fully exploits the spatial-temporal information among video tokens and prompt tokens


## Overview
<p align="center">
<img src="assets/ump_tab.png" width=100% height=100% 
class="center">
</p>

Prompt tuning, an emerging parameter-efficient strategy, leverages the powerful knowledge of large-scale pre-trained image-text models (<i>e.g.</i>, CLIP) to swiftly adapt to downstream tasks. Despite its effectiveness, adapting prompt tuning to text-video retrieval encounters two limitations: i) existing methods adopt two isolated prompt tokens to prompt two modal branches separately, making it challenging to learn a well-aligned unified representation, <i>i.e.</i>, modality gap; ii) video encoders typically utilize a fixed pre-trained visual backbone, neglecting the incorporation of spatial-temporal information. To this end, we propose a simple yet effective method, named Unified Modality-aware Prompt Tuning (UMP), for text-video retrieval. Concretely, we first introduce a Unified Prompt Generation (UPG) module to dynamically produce modality-aware prompt tokens, enabling the perception of prior semantic information on both video and text inputs. These prompt tokens are simultaneously injected into two branches that can bridge the semantics gap between two modalities in a unified-adjusting manner. Then, we design a parameter-free Spatial-Temporal Shift (STS) module to facilitate both intra- and inter-communication among video tokens and prompt tokens in the spatial-temporal dimension. Notably, extensive experiments on four widely used benchmarks show that UMP achieves new state-of-the-art performance compared to existing prompt-tuning methods without bringing excessive parameters.

## Usage


## Bibliography
If you find this repository helpful for your project, please consider citing our work:

```
@article{zhang2024ump,
  title={UMP: Unified Modality-aware Prompt Tuning for Text-Video Retrieval},
  author={Haonan Zhang, Pengpeng Zeng, Lianli Gao, Jingkuan Song, Heng Tao Shen},
  journal={arXiv preprint arXiv:XXX.XXX},
  year={2024}
}
```
