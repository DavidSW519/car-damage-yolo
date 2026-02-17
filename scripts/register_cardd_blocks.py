from ultralytics.nn import tasks
from ultralytics.nn.modules.cardd_blocks import ResBlock, ConvNeXtBlockLite, MBV3BlockLite

# Make parse_model globals() find these names
tasks.ResBlock = ResBlock
tasks.ConvNeXtBlockLite = ConvNeXtBlockLite
tasks.MBV3BlockLite = MBV3BlockLite

print("✅ Registered CarDD blocks into ultralytics.nn.tasks")
