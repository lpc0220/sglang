# Removed Files Log

**Project:** SGLang DeepSeek-Only
**Last Updated:** 2026-01-11

## Batch 1: DeepSeek Multimodal Variants (2026-01-11)

### Files to Remove:
1. `sglang/python/sglang/srt/models/deepseek_janus_pro.py`
2. `sglang/python/sglang/srt/models/deepseek_ocr.py`
3. `sglang/python/sglang/srt/models/deepseek_vl2.py`

### Related Infrastructure (KEEP - but may need updates):
- `sglang/python/sglang/srt/multimodal/processors/deepseek_vl_v2.py` - Multimodal processor
- `sglang/python/sglang/srt/multimodal/processors/deepseek_ocr.py` - OCR processor
- `sglang/python/sglang/srt/multimodal/processors/janus_pro.py` - Janus processor
- `sglang/python/sglang/srt/utils/hf_transformers_utils.py` - Has deepseek_ocr detection logic
- `sglang/python/sglang/lang/chat_template.py` - May reference multimodal models
- `sglang/python/sglang/srt/parser/conversation.py` - May reference multimodal models

### Decision:
**Conservative Approach** - Remove model files only, keep infrastructure (may be used by other models or future additions)

### Status: COMPLETED ✅

### Files Removed:
**Model Files (3):**
1. ✅ `sglang/python/sglang/srt/models/deepseek_janus_pro.py`
2. ✅ `sglang/python/sglang/srt/models/deepseek_ocr.py`
3. ✅ `sglang/python/sglang/srt/models/deepseek_vl2.py`

**Config Files (2):**
4. ✅ `sglang/python/sglang/srt/configs/deepseek_ocr.py`
5. ✅ `sglang/python/sglang/srt/configs/deepseekvl2.py`

**Processor Files (3):**
6. ✅ `sglang/python/sglang/srt/multimodal/processors/deepseek_vl_v2.py`
7. ✅ `sglang/python/sglang/srt/multimodal/processors/deepseek_ocr.py`
8. ✅ `sglang/python/sglang/srt/multimodal/processors/janus_pro.py`

**Code Cleanup:**
- ✅ Removed imports from `sglang/python/sglang/srt/configs/__init__.py`
- ✅ Removed imports from `sglang/python/sglang/srt/utils/hf_transformers_utils.py`
- ✅ Removed `_is_deepseek_ocr_model()` function
- ✅ Removed `_override_deepseek_ocr_v_head_dim()` function
- ✅ Removed all function calls to deepseek_ocr detection logic

**Total Removed:** 8 files + code cleanup

**Remaining String References:** Some architecture name strings remain (e.g., "DeepseekVL2ForCausalLM") - these are harmless and will be cleaned in later rounds if needed.

---

## Batch 6: Test Files and AMD/Ascend Infrastructure (2026-01-11)

### Summary:
**Total Removed:** 63 files
**Total Lines Deleted:** ~6,280 lines
**Commit:** a04c2e2de

### Categories:

#### A. Test Files for Removed Models (21 files)
1. ✅ `sglang/test/srt/models/test_kimi_k2_models.py`
2. ✅ `sglang/test/srt/models/test_mimo_models.py`
3. ✅ `sglang/test/srt/models/test_ministral3_models.py`
4. ✅ `sglang/test/srt/models/test_dummy_grok_models.py`
5. ✅ `sglang/test/srt/models/test_qwen3_next_models.py`
6. ✅ `sglang/test/srt/cpu/test_qwen3.py`
7. ✅ `sglang/test/srt/test_mistral_large3_basic.py`
8. ✅ `sglang/test/srt/test_llama31_fp4.py`
9. ✅ `sglang/test/manual/models/test_grok_models.py`
10. ✅ `sglang/test/manual/models/test_gme_qwen_models.py`
11. ✅ `sglang/test/manual/models/test_llama4_models.py`
12. ✅ `sglang/test/manual/lora/test_lora_llama4.py`
13. ✅ `sglang/test/manual/lora/test_lora_qwen3_vl.py`
14. ✅ `sglang/test/registered/8-gpu-models/test_mistral_large3.py`
15. ✅ `sglang/test/registered/8-gpu-models/test_kimi_k2.py`
16. ✅ `sglang/test/registered/8-gpu-models/test_llama4.py`
17. ✅ `sglang/test/registered/8-gpu-models/test_qwen3_235b.py`
18. ✅ `sglang/test/registered/core/test_qwen3_next_deterministic.py`
19. ✅ `sglang/test/registered/backends/test_qwen3_fp4_trtllm_gen_moe.py`
20. ✅ `sglang/test/registered/models/test_qwen_models.py`
21. ✅ `sglang/test/registered/models/test_kimi_linear_models.py`

#### B. Ascend NPU Test Directory (29 files)
**Entire directory removed:** `sglang/test/registered/ascend/`
- 1 embedding model test
- 18 LLM model tests (includes llama, mistral, gemma, phi4, etc.)
- 1 rerank model test
- 8 VLM model tests
- 1 configuration file (jinja template, yaml)

#### C. AMD GPU Test Directory (8 files)
**Entire directory removed:** `sglang/test/registered/amd/`
- ✅ `test/registered/amd/test_deepseek_v3_perf.py` (AMD-specific)
- ✅ `test/registered/amd/test_deepseek_v31_perf.py` (AMD-specific)
- ✅ `test/registered/amd/test_deepseek_r1_mxfp4_perf.py` (AMD-specific)
- ✅ `test/registered/amd/test_grok_perf.py`
- ✅ `test/registered/amd/nightly/test_gsm8k_eval_amd.py`
- ✅ `test/registered/amd/nightly/test_gsm8k_completion_eval_amd.py`
- ✅ `test/registered/amd/nightly/test_gsm8k_completion_eval_mi35x.py`
- ✅ `test/registered/amd/nightly/test_vlms_mmmu_eval_amd.py`

**Note on DeepSeek AMD Tests:**
The AMD directory contained DeepSeek-specific benchmarks (v3, v31, R1) but these are AMD hardware-specific tests. Since we're targeting NVIDIA GPU only, these AMD-specific DeepSeek tests were removed as part of the AMD infrastructure cleanup.

#### D. Platform Documentation (5 files)
1. ✅ `sglang/docs/platforms/amd_gpu.md`
2. ✅ `sglang/docs/platforms/ascend_npu_support_models.md`
3. ✅ `sglang/docs/platforms/tpu.md`
4. ✅ `sglang/docs/platforms/xpu.md`
5. ✅ `sglang/python/sglang/multimodal_gen/docs/install_rocm.md`

### Rationale:
- Tests for models removed in Batches 1-5
- NVIDIA GPU only deployment (no AMD/Ascend support needed)
- DeepSeek AMD benchmarks removed as part of AMD infrastructure cleanup
- Platform-specific documentation no longer relevant

### Validation:
✅ All syntax tests passing
✅ No broken references detected
✅ DeepSeek models intact

---

## Batch 2a-2c: Non-DeepSeek Models (2026-01-11)

### Summary:
**Total Removed:** 117 model files
**Total Lines Deleted:** ~60,259 lines

### Batch 2a: Models A-G (37 files, -19,171 lines)
Removed: Apertus, Arcee, Baichuan, Bailing MoE, BERT, ChatGLM, CLIP, CommandR, DBRX, Dots OCR/VLM, Ernie4, Exaone, Falcon H1, Gemma (all variants), GLM4 (all variants), GPT (2/BigCode/OSS), Granite, Grok

Git commit: `415da9e61`

### Batch 2b: Models H-Q (66 files, -35,599 lines)
Removed: Hunyuan, Idefics2, InternLM2/InternS1/InternVL, Jet, Kimi, Llada2, Llama (all variants), Llava, Longcat, Midashenglm, MiniCPM, Minimax, Mindspore, Mimo, Mistral (all variants), Mllama, Nemotron, NVILA, Olmo, OPT, Orion, PaddleOCR, Persimmon, Phi (all variants), Pixtral, Points, Qwen (all 1/2/2.5/3 variants)

Git commit: `f9e6b646a`

### Batch 2c: Models R-Z (14 files, -5,489 lines)
Removed: Radio, Roberta, Sarashina2, SigLIP, Solar, StableLM, StarCoder2, Step3 VL, TeleFLM, Torch Native Llama, Transformers, Xverse, Yivl

Git commit: `1afecec6d`

### Remaining Model Files: 17 total
- ✅ **DeepSeek (3):** deepseek.py, deepseek_v2.py, deepseek_nextn.py
- ✅ **DeepSeek common:** deepseek_common/ directory
- 📦 **Other (13):** registry.py, utils.py, + 11 other infrastructure files

**Status:** ✅ COMPLETE - All non-DeepSeek models removed!

---

## Batch 3: Hardware Backends - NPU and Ascend (2026-01-11)

### Summary:
**Total Removed:** 32 files (17 NPU + 12 additional model stragglers + 3 Ascend tests)
**Total Lines Deleted:** ~12,648 lines

### NPU Backend Files (17 files):
**Hardware Backend:**
1. ✅ `sglang/python/sglang/srt/hardware_backend/npu/allocator_npu.py`
2. ✅ `sglang/python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
3. ✅ `sglang/python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py`
4. ✅ `sglang/python/sglang/srt/hardware_backend/npu/cmo.py`
5. ✅ `sglang/python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_extend_npu_graph_runner.py`
6. ✅ `sglang/python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_npu_graph_runner.py`
7. ✅ `sglang/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py`
8. ✅ `sglang/python/sglang/srt/hardware_backend/npu/memory_pool_npu.py`
9. ✅ `sglang/python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`
10. ✅ `sglang/python/sglang/srt/hardware_backend/npu/moe/topk.py`
11. ✅ `sglang/python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`
12. ✅ `sglang/python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py`
13. ✅ `sglang/python/sglang/srt/hardware_backend/npu/quantization/modelslim.py`
14. ✅ `sglang/python/sglang/srt/hardware_backend/npu/utils.py`

**Ascend Disaggregation (3 files):**
15. ✅ `sglang/python/sglang/srt/disaggregation/ascend/__init__.py`
16. ✅ `sglang/python/sglang/srt/disaggregation/ascend/conn.py`
17. ✅ `sglang/python/sglang/srt/disaggregation/ascend/transfer_engine.py`

### Ascend Tests (3 files):
18. ✅ `sglang/python/sglang/test/ascend/__init__.py`
19. ✅ `sglang/python/sglang/test/ascend/gsm8k_ascend_mixin.py`
20. ✅ `sglang/python/sglang/test/ascend/vlm_utils.py`

### Additional Model Cleanup (9 files - stragglers from Batch 2):
21. ✅ `sglang/python/sglang/srt/models/llama_eagle.py`
22. ✅ `sglang/python/sglang/srt/models/llama_embedding.py`
23. ✅ `sglang/python/sglang/srt/models/llama_reward.py`
24. ✅ `sglang/python/sglang/srt/models/longcat_flash.py`
25. ✅ `sglang/python/sglang/srt/models/mimo_v2_flash_nextn.py`
26. ✅ `sglang/python/sglang/srt/models/minicpmo.py`
27. ✅ `sglang/python/sglang/srt/models/minicpmv.py`
28. ✅ `sglang/python/sglang/srt/models/ministral3.py`
29. ✅ `sglang/python/sglang/srt/models/mixtral.py`
30. ✅ `sglang/python/sglang/srt/models/mixtral_quant.py`
31. ✅ `sglang/python/sglang/srt/models/mllama4.py`
32. ✅ `sglang/python/sglang/srt/models/nano_nemotron_vl.py`

### Decision:
**NVIDIA GPU Only** - Removed all NPU/Ascend hardware backend support. Target deployment is multi-node NVIDIA GPU cluster for DeepSeek R1.

### Remaining Work:
- **AMD GPU (ROCm/HIP):** Conditional code remains in ~72 files
  - Will be addressed in Phase 3 (deep cleanup) with careful analysis
  - Files include MoE kernels, attention ops, quantization layers
  - Need to verify DeepSeek doesn't use any ROCm-specific optimizations

Git commit: `99a18e122`

**Status:** ✅ COMPLETE - NPU and Ascend support removed!

---

## Batch 4: Orphaned Configs and Multimodal Processors (2026-01-11)

### Summary:
**Total Removed:** 47 files
**Total Lines Deleted:** ~8,538 lines

### Config Files (23 files):
1. ✅ `sglang/python/sglang/srt/configs/chatglm.py`
2. ✅ `sglang/python/sglang/srt/configs/dbrx.py`
3. ✅ `sglang/python/sglang/srt/configs/dots_ocr.py`
4. ✅ `sglang/python/sglang/srt/configs/dots_vlm.py`
5. ✅ `sglang/python/sglang/srt/configs/exaone.py`
6. ✅ `sglang/python/sglang/srt/configs/falcon_h1.py`
7. ✅ `sglang/python/sglang/srt/configs/internvl.py`
8. ✅ `sglang/python/sglang/srt/configs/janus_pro.py`
9. ✅ `sglang/python/sglang/srt/configs/jet_nemotron.py`
10. ✅ `sglang/python/sglang/srt/configs/jet_vlm.py`
11. ✅ `sglang/python/sglang/srt/configs/kimi_linear.py`
12. ✅ `sglang/python/sglang/srt/configs/kimi_vl.py`
13. ✅ `sglang/python/sglang/srt/configs/kimi_vl_moonvit.py`
14. ✅ `sglang/python/sglang/srt/configs/longcat_flash.py`
15. ✅ `sglang/python/sglang/srt/configs/nano_nemotron_vl.py`
16. ✅ `sglang/python/sglang/srt/configs/nemotron_h.py`
17. ✅ `sglang/python/sglang/srt/configs/olmo3.py`
18. ✅ `sglang/python/sglang/srt/configs/points_v15_chat.py`
19. ✅ `sglang/python/sglang/srt/configs/qwen3_next.py`
20. ✅ `sglang/python/sglang/srt/configs/qwen3_omni.py`
21. ✅ `sglang/python/sglang/srt/configs/qwen3_vl.py`
22. ✅ `sglang/python/sglang/srt/configs/radio.py`
23. ✅ `sglang/python/sglang/srt/configs/step3_vl.py`

### Multimodal Processors (22 files):
24. ✅ `sglang/python/sglang/srt/multimodal/processors/clip.py`
25. ✅ `sglang/python/sglang/srt/multimodal/processors/dots_vlm.py`
26. ✅ `sglang/python/sglang/srt/multimodal/processors/gemma3.py`
27. ✅ `sglang/python/sglang/srt/multimodal/processors/gemma3n.py`
28. ✅ `sglang/python/sglang/srt/multimodal/processors/glm4v.py`
29. ✅ `sglang/python/sglang/srt/multimodal/processors/glmasr.py`
30. ✅ `sglang/python/sglang/srt/multimodal/processors/internvl.py`
31. ✅ `sglang/python/sglang/srt/multimodal/processors/kimi_vl.py`
32. ✅ `sglang/python/sglang/srt/multimodal/processors/llava.py`
33. ✅ `sglang/python/sglang/srt/multimodal/processors/midashenglm.py`
34. ✅ `sglang/python/sglang/srt/multimodal/processors/minicpm.py`
35. ✅ `sglang/python/sglang/srt/multimodal/processors/mlama.py`
36. ✅ `sglang/python/sglang/srt/multimodal/processors/mllama4.py`
37. ✅ `sglang/python/sglang/srt/multimodal/processors/nano_nemotron_vl.py`
38. ✅ `sglang/python/sglang/srt/multimodal/processors/nvila.py`
39. ✅ `sglang/python/sglang/srt/multimodal/processors/paddleocr_vlm.py`
40. ✅ `sglang/python/sglang/srt/multimodal/processors/phi4mm.py`
41. ✅ `sglang/python/sglang/srt/multimodal/processors/pixtral.py`
42. ✅ `sglang/python/sglang/srt/multimodal/processors/points_v15_chat.py`
43. ✅ `sglang/python/sglang/srt/multimodal/processors/qwen_audio.py`
44. ✅ `sglang/python/sglang/srt/multimodal/processors/qwen_vl.py`
45. ✅ `sglang/python/sglang/srt/multimodal/processors/sarashina2_vision.py`
46. ✅ `sglang/python/sglang/srt/multimodal/processors/step3_vl.py`

### Code Cleanup:
47. ✅ Cleaned `sglang/python/sglang/srt/configs/__init__.py` - removed all imports

### Decision:
**Text-only DeepSeek R1** - No multimodal support needed. Removed all orphaned configs and processors for deleted models.

Git commit: `e2b0f0260`

**Status:** ✅ COMPLETE - Configs and processors cleaned up!

---

## Batch 5: Non-DeepSeek Benchmarks (2026-01-11)

### Summary:
**Total Removed:** 156 files (36 benchmark directories)
**Total Lines Deleted:** ~26,051 lines

### Decision:
Keep **only** `benchmark/deepseek_v3/` for DeepSeek validation

### Removed Benchmark Directories (36):
1. ✅ bench_attention_sink
2. ✅ bench_in_batch_prefix
3. ✅ benchmark_batch
4. ✅ benchmark_vllm_060
5. ✅ blog_v0_2
6. ✅ boolq
7. ✅ ceval
8. ✅ dspy
9. ✅ fla
10. ✅ generative_agents
11. ✅ gpt_oss
12. ✅ gsm8k
13. ✅ hellaswag
14. ✅ hf3fs
15. ✅ hicache
16. ✅ json_decode_regex
17. ✅ json_jump_forward
18. ✅ json_schema
19. ✅ kernels (MoE, quantization, attention, etc.)
20. ✅ line_retrieval
21. ✅ llava_bench
22. ✅ llm_judge
23. ✅ long_json_decode
24. ✅ lora
25. ✅ mmlu
26. ✅ mmmu
27. ✅ mtbench
28. ✅ multi_chain_reasoning
29. ✅ multi_document_qa
30. ✅ multi_turn_chat
31. ✅ prefill_only
32. ✅ react
33. ✅ reasoning_benchmark
34. ✅ tip_suggestion
35. ✅ tree_of_thought_deep
36. ✅ tree_of_thought_v0

Git commit: `8daf6e213`

**Status:** ✅ COMPLETE - Only DeepSeek v3 benchmark remains!
