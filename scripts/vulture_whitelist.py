# Vulture whitelist: false positives for unused parameters.
#
# PyTorch register_forward_hook callbacks require (module, input, output)
# signature even when the callback only uses output.
module
input

# Mechanistic analysis parameter reserved for future attention-head ablation.
skip_heads
