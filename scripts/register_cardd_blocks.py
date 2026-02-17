from ultralytics.nn import tasks
from scripts.cardd_blocks import ResBlock, ConvNeXtBlockLite, MBV3BlockLite

tasks.__dict__.update({
    "ResBlock": ResBlock,
    "ConvNeXtBlockLite": ConvNeXtBlockLite,
    "MBV3BlockLite": MBV3BlockLite,
})

print("✅ Registered CarDD blocks into ultralytics.nn.tasks")
