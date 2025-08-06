---
title: "Vibe checking GPT-OSS with vLLM, Modal, and Textual"
date: 2025-08-06T00:00:00-04:00
draft: false
tags: ["vllm", "modal", "textual", "gpt-oss", "openai", "technical"]
params:
  author: "jbohnslav"
---

OpenAI's just open-sourced their first model since GPT2. I wanted to try out a few prompts, check the vibes, and see OpenAI's raw reasoning traces for the first time. Alas, [gpt-oss.com](http://gpt-oss.com) was down for the whole day. Inference providers weren't up yet. My old code didn't use the [Responses API](https://platform.openai.com/docs/api-reference/responses), so I couldn't see the reasoning trace or change the reasoning effort.

So, I basically just smashed five blog posts together to make a Modal vLLM server and a Textual Python client where we can chat with GPT-OSS-120b!

{{< video src="/videos/textual_gptoss.mp4" width="100%" >}}{{< /video >}}

> Check out the code **[on my GitHub](https://github.com/jbohnslav/modal_chat).**

## Running GPT-OSS on Modal

Modal gives me $30 per month of free GPUs and spins up and down quickly. This seemed ideal for my "try out GPT-OSS-120B for a few minutes" application.

The key bits [of the modal vllm server script](https://github.com/jbohnslav/modal_chat/blob/main/vllm_server.py) are setting up the server with Modal's docker-like syntax:

```python
vllm_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .entrypoint([])  # remove verbose logging by base image on entry
    .uv_pip_install(
        "vllm==0.10.1+gptoss",
        extra_index_url="https://wheels.vllm.ai/gpt-oss/",
        pre=True,
        extra_options="--extra-index-url https://download.pytorch.org/whl/nightly/cu128 --index-strategy unsafe-best-match",
    )
    .uv_pip_install("huggingface_hub[hf_transfer]")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_USE_V1": "1",
            "TORCH_CUDA_ARCH_LIST": "9.0;10.0",  # H100/H200 (9.0) and B200 (10.0)
        }
    )
)
```

and running a subprocess call but within a modal `app.function` and `modal.web_server`, which will give us a public API we can use. A note on the memory snapshot: I experimented with this to try to reduce my cold-start time, but I haven't been able to get it to work yet.

```python
# Create app
app = modal.App("vllm-server")

# Serve function
@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    scaledown_window=15 * 60,
    timeout=10 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    enable_memory_snapshot=True, # I don't think this works properly yet
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=2)
@modal.web_server(port=8000, startup_timeout=10 * 60)
def serve() -> None:
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--tensor-parallel-size",
        str(N_GPU),
        "--max-model-len",
        str(MAX_MODEL_LEN),
    ]

    if MODEL_REVISION:
        cmd.extend(["--revision", MODEL_REVISION])

    subprocess.Popen(" ".join(cmd), shell=True)
```

**Note:** This will give you a public link with no authentication. If you don't want randos to use your GPU, you at least set `VLLM_API_KEY` in both the server and client.

Click the resulting link so that you can see the logs in Modal's beautiful viewer. Here's mine.

![Modal logs screenshot](/images/modal_ss.png)

You can see vLLM spinning up, loading safetensors, etc. After a few minutes (could've been faster if I'd disabled cuda graphs and torch.compile), you'll see

```bash
(APIServer pid=5) INFO:     Application startup complete.
```

Test with standard OpenAI API calls:

```bash
export OPENAI_BASE_URL=https://{your-modal-username}--vllm-server-serve.modal.run/v1
curl $OPENAI_BASE_URL/models

# you should see the below
{"object":"list","data":[{"id":"openai/gpt-oss-120b","object":"model","created":1754450540,"owned_by":"vllm","root":"openai/gpt-oss-120b","parent":null,"max_model_len":65536,"permission":[{"id":"modelperm-4746fc4aa9a2413b905e840e9be92ac9","object":"model_permission","created":1754450540,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]
```

Groovy!

## Using Textual to create a minimalist chat app with OpenAI's new Responses API

The most interesting thing about GPT-OSS is that it's the first time we have unrestricted access to an OpenAI model's raw chains of thought. With the much more common Completions API, I realized I wasn't seeing the actual reasoning trace. I also couldn't switch between low, medium, and high reasoning. The newer [Responses API](https://platform.openai.com/docs/api-reference/responses) lets us do that. So, I more-or-less put Will McGugan's [Mother AI example code](https://textual.textualize.io/blog/2024/09/15/anatomy-of-a-textual-user-interface/) into Claude Code and told it to

- Update from completions to the Responses API
- Make sure streaming works
- Give nice boxes around my prompt, the reasoning trace, and the final answer separately.

The hardest part was debugging Textual's streaming markdown handling, but after a few hours, it's up and running. It even looks kind of nice!

![Textual UI screenshot](/images/textual_ui.png)

Now that we have the vibe-tooling, the vibe-checking can begin.

## References

- [Modal docs: Run OpenAI-compatible LLM inference with LLaMA 3.1-8B and vLLM](https://modal.com/docs/examples/vllm_inference)
- [Modal:GPU memory snapshots](https://modal.com/blog/gpu-mem-snapshots)
- [Mother app: example Textual UI from Will McGugan](https://textual.textualize.io/blog/2024/09/15/anatomy-of-a-textual-user-interface/)
- [OpenAI: How to run gpt-oss with vLLM](https://cookbook.openai.com/articles/gpt-oss/run-vllm)
- [vLLM now supports gpt-oss](https://blog.vllm.ai/2025/08/05/gpt-oss.html)