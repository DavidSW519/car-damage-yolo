from ultralytics.nn import tasks
from ultralytics.nn.modules.cardd_backbones import (
    ResNetStem, ResNetStage,
    ConvNeXtStem, ConvNeXtStage,
    MobileNetV3Stem, MobileNetV3Stage,
)

# Inject into ultralytics.nn.tasks namespace (globals() lookup)
tasks.ResNetStem = ResNetStem
tasks.ResNetStage = ResNetStage
tasks.ConvNeXtStem = ConvNeXtStem
tasks.ConvNeXtStage = ConvNeXtStage
tasks.MobileNetV3Stem = MobileNetV3Stem
tasks.MobileNetV3Stage = MobileNetV3Stage

# Ensure parse_model() treats these as base modules and infers c2 from args[0]
bm = getattr(tasks, "base_modules", ())
if isinstance(bm, tuple):
    tasks.base_modules = bm + (ResNetStem, ResNetStage, ConvNeXtStem, ConvNeXtStage, MobileNetV3Stem, MobileNetV3Stage)

print("✅ Registered CarDD custom modules into ultralytics.nn.tasks (+ base_modules)")
