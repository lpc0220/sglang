"""
ModelOpt related constants
DeepSeek R1 uses FP4/FP8 only - AWQ configurations removed
"""

QUANT_CFG_CHOICES = {
    "fp8": "FP8_DEFAULT_CFG",
    # REMOVED: "int4_awq": "INT4_AWQ_CFG" - DeepSeek R1 uses FP4/FP8 only
    # REMOVED: "w4a8_awq": "W4A8_AWQ_BETA_CFG" - DeepSeek R1 uses FP4/FP8 only
    "nvfp4": "NVFP4_DEFAULT_CFG",
    # REMOVED: "nvfp4_awq": "NVFP4_AWQ_LITE_CFG" - DeepSeek R1 uses FP4/FP8 only
}
