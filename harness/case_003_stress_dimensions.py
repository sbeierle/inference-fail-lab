import torch
from case_003_triton_standalone import test_block_size

print("\n" + "="*75)
print("STRESSTEST: PROBLEMATISCHE HEAD-DIMENSIONEN (96, 192, 256)")
print("="*75)

# Teste ungerade und große Head-Dimensionen aus Issue #62697
for h in [96, 192, 256]:
    for b in [16, 64, 528]:
        test_block_size(head_size=h, block_size=b)

print("="*75)
