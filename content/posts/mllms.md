---
title: "MLLMs, VLMs, LVLMs, LMMs..."
description: "What do we call these things?"
date: 2024-12-11T21:11:57-05:00
draft: false
tags: ["mllm", "vlm", "lvlm", "lmm"]
---



There exists a class of models whose *inputs are text prompts + images or video. Their outputs are text.* 

Example:  "Explain the joke in this tweet. Be concise."
{{< figure src="/images/server-joke.webp" alt="home-server-joke" width="600" >}}

Answer, courtesy of GPT4o:
> The joke humorously compares "the talk" about sensitive topics with explaining to kids why there's a server at home. The mock children's book title exaggerates the idea, poking fun at tech enthusiasts whose home servers are significant enough to require a formal explanation to their kids.

Building these models is one of the biggest fields of both industrial AI and academic computer vision research. But no one can agree on what to call them! For the rest of this post I'm going to keep calling them "these models" because I don't know what else to do.

## Choices

**Multimodal Large Language Models (MLLM)**

This is probably the most common name for "these models". Points in favor: they all use LLMs as a core component. These models *generate text*, just like LLMs. They are multi-modal, able to process multiple types of inputs.

The problem is that this name is not specific enough: imagine a model where you submit an audio file to and ask a question, e.g. "What sound is this?"  + `sound.wav` -> "This is a siren of an emergency vehicle." Such a model would also be an MLLM. 

Proponents: [vLLM](https://docs.vllm.ai/en/latest/usage/multimodal_inputs.html#multimodal-inputs), Llama 3[^1], Pixtral[^2], Gemini[^3], Waymo[^4], Cambrian-1[^5], InternVL2.5 [^6], Mammoth-VL[^7], Florence-VL [^8], Fei Fei Li (NeurIPS)

**Vision-language model (VLM)**

VLM is more specific than MLLM, which is good. However, models like CLIP and SigLIP are Vision-Language Models too. They have image encoders, text encoders, can be prompted, etc. But CLIP et al. are not *generative*; they do not produce text. That makes this term confusing to me.  

Proponents: Molmo[^9], [Huggingface (SmolVLM)](https://huggingface.co/blog/smolvlm), PaliGemma 2 [^10], CogVLM[^11], NVILA[^12]

**Large vision-language model (LVLM)**

"These models" are large, use vision, and generate language. Pretty good. But InternViT-6B is a CLIP-style model with 6 billion parameters: it's large by any measure. InternViT isn't generative, so it's not the kind of model I mean. This paper[^16] even calls CLIP a VLM and "these models" LVLMs, so I guess generating text is what makes it "Large"?

Proponents: Qwen2-VL[^13]

**Large Multimodal Model (LMM)**

This one is popular with the Llava folks. They get extra credit because their paper defined the field, but I see this as just a variant of MLLM. 

Proponents: Llava[^14], Llava-OneVision[^15]

## My take

I've been a proponent of calling "these models" MLLMs. However, models like GPT4o and Gemini Flash 2.0 can consume text, images, video, or audio, and generate text, images, or audio as well. *That* is truly multimodal. It's such a big difference that the GPT4o system card[^17] calls it an "omni model". 

Models focusing on images and videos specifically are going to be extremely valuable in many domains: robotics, web agents, as components in coding assistants, and in consumer apps. It therefore makes sense to define them as a class distinctly from the "omni models." 

Through writing this post, I've convinced myself that VLM is a more specific, useful term. With great apologies to Lucas Beyer and the rest of the SigLIP team, I will call models that learn a joint embedding space between images and text "CLIP-style models."

What do you think we should call VLMs? Let's discuss on Twitter or BlueSky. 

## References

[^1]: A. Dubey _et al._, "The Llama 3 Herd of Models," Jul. 31, 2024, _arXiv_: arXiv:2407.21783. Accessed: Aug. 01, 2024. [Online]. Available: [http://arxiv.org/abs/2407.21783](http://arxiv.org/abs/2407.21783)

[^2]: P. Agrawal _et al._, "Pixtral 12B," Oct. 09, 2024, _arXiv_: arXiv:2410.07073. Accessed: Oct. 10, 2024. [Online]. Available: [http://arxiv.org/abs/2410.07073](http://arxiv.org/abs/2410.07073)

[^3]: G. Team _et al._, "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context," Aug. 08, 2024, _arXiv_: arXiv:2403.05530. doi: [10.48550/arXiv.2403.05530](https://doi.org/10.48550/arXiv.2403.05530).

[^4]: J.-J. Hwang _et al._, "EMMA: End-to-End Multimodal Model for Autonomous Driving," Oct. 30, 2024, _arXiv_: arXiv:2410.23262. Accessed: Nov. 04, 2024. [Online]. Available: [http://arxiv.org/abs/2410.23262](http://arxiv.org/abs/2410.23262)

[^5]: S. Tong _et al._, "Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs," Jun. 24, 2024, _arXiv_: arXiv:2406.16860. Accessed: Jun. 25, 2024. [Online]. Available: [http://arxiv.org/abs/2406.16860](http://arxiv.org/abs/2406.16860)

[^6]: Z. Chen _et al._, "Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling," Dec. 06, 2024, _arXiv_: arXiv:2412.05271. doi: [10.48550/arXiv.2412.05271](https://doi.org/10.48550/arXiv.2412.05271).

[^7]: J. Guo _et al._, "MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale," Dec. 06, 2024, _arXiv_: arXiv:2412.05237. doi: [10.48550/arXiv.2412.05237](https://doi.org/10.48550/arXiv.2412.05237).

[^8]: J. Chen _et al._, "Florence-VL: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion," Dec. 05, 2024, _arXiv_: arXiv:2412.04424. doi: [10.48550/arXiv.2412.04424](https://doi.org/10.48550/arXiv.2412.04424).

[^9]: M. Deitke _et al._, "Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models," Sep. 25, 2024, _arXiv_: arXiv:2409.17146. Accessed: Sep. 26, 2024. [Online]. Available: [http://arxiv.org/abs/2409.17146](http://arxiv.org/abs/2409.17146)

[^10]: A. Steiner _et al._, "PaliGemma 2: A Family of Versatile VLMs for Transfer," Dec. 04, 2024, _arXiv_: arXiv:2412.03555. doi: [10.48550/arXiv.2412.03555](https://doi.org/10.48550/arXiv.2412.03555).

[^11]: W. Hong _et al._, "CogVLM2: Visual Language Models for Image and Video Understanding," Aug. 29, 2024, _arXiv_: arXiv:2408.16500. Accessed: Aug. 30, 2024. [Online]. Available: [http://arxiv.org/abs/2408.16500](http://arxiv.org/abs/2408.16500)

[^12]: Z. Liu _et al._, "NVILA: Efficient Frontier Visual Language Models," Dec. 05, 2024, _arXiv_: arXiv:2412.04468. doi: [10.48550/arXiv.2412.04468](https://doi.org/10.48550/arXiv.2412.04468).

[^13]: P. Wang _et al._, "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution," Sep. 18, 2024, _arXiv_: arXiv:2409.12191. Accessed: Sep. 19, 2024. [Online]. Available: [http://arxiv.org/abs/2409.12191](http://arxiv.org/abs/2409.12191)

[^14]: H. Liu, C. Li, Q. Wu, and Y. J. Lee, "Visual Instruction Tuning," Dec. 11, 2023, _arXiv_: arXiv:2304.08485. Accessed: Jun. 28, 2024. [Online]. Available: [http://arxiv.org/abs/2304.08485](http://arxiv.org/abs/2304.08485)

[^15]: B. Li _et al._, "LLaVA-OneVision: Easy Visual Task Transfer," Aug. 06, 2024, _arXiv_: arXiv:2408.03326. Accessed: Aug. 07, 2024. [Online]. Available: [http://arxiv.org/abs/2408.03326](http://arxiv.org/abs/2408.03326)

[^16]: Y. Ouali _et al._, "Discriminative Fine-tuning of LVLMs," Dec. 05, 2024, _arXiv_: arXiv:2412.04378. doi: [10.48550/arXiv.2412.04378](https://doi.org/10.48550/arXiv.2412.04378).

[^17]: 