"""
Patches LeRobot factory.py to support smolvla_moh policy type.
Run before any lerobot-train command.
"""
import sys
import os

# Find LeRobot factory path
import lerobot
lerobot_dir = os.path.dirname(os.path.dirname(lerobot.__file__))
factory_path = os.path.join(lerobot_dir, 'lerobot/policies/factory.py')

print(f"Patching: {factory_path}")

with open(factory_path, 'r') as f:
    content = f.read()

eager_import = '''import sys as _sys, os as _os
_sys.path.insert(0, _os.path.expanduser("~/smolvla-moh-project"))
from models.smolvla_moh.moh_config import SmolVLAMoHConfig  # registers smolvla_moh subclass
'''

already_eager = 'SmolVLAMoHConfig  # registers smolvla_moh subclass' in content
already_lazy  = 'smolvla_moh' in content and 'SmolVLAMoHPolicy' in content

if already_eager and already_lazy:
    print('Already patched')
    sys.exit(0)

# Patch get_policy_class
old_smolvla = '    elif name == "smolvla":\n        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy\n\n        return SmolVLAPolicy'
new_smolvla = '''    elif name == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        return SmolVLAPolicy
    elif name == "smolvla_moh":
        import sys, os
        sys.path.insert(0, os.path.expanduser("~/smolvla-moh-project"))
        from models.smolvla_moh.moh_policy import SmolVLAMoHPolicy
        return SmolVLAMoHPolicy'''

# Patch make_policy_config
old_config = '    elif policy_type == "smolvla":\n'
new_config = '''    elif policy_type == "smolvla_moh":
        import sys, os
        sys.path.insert(0, os.path.expanduser("~/smolvla-moh-project"))
        from models.smolvla_moh.moh_config import SmolVLAMoHConfig
        return SmolVLAMoHConfig(**kwargs)
    elif policy_type == "smolvla":
'''

if not already_eager:
    # Insert eager import right after the SmolVLAConfig import so the
    # @PreTrainedConfig.register_subclass("smolvla_moh") decorator runs at
    # module load time — this is what populates the argparse choices list.
    content = content.replace(
        'from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig\n',
        'from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig\n' + eager_import,
    )

if not already_lazy:
    content = content.replace(old_smolvla, new_smolvla)
    content = content.replace(old_config, new_config)

with open(factory_path, 'w') as f:
    f.write(content)

print('Factory patched successfully')

# Verify
from importlib import reload
import lerobot.policies.factory as factory_module
reload(factory_module)
from lerobot.policies.factory import get_policy_class
cls = get_policy_class('smolvla_moh')
print(f'Verified: {cls}')