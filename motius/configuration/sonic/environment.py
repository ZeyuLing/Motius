"""Released SONIC G1 manager configuration, resolved to native components."""

from pathlib import Path
import importlib
import xml.etree.ElementTree as ET


def instantiate(configuration):
    """Construct nested SDK/MDP configuration objects without a trainer framework."""
    if isinstance(configuration, list):
        return [instantiate(value) for value in configuration]
    if not isinstance(configuration, dict):
        return configuration
    values = {
        key: instantiate(value)
        for key, value in configuration.items()
        if not key.startswith("_")
    }
    target = configuration.get("_target_")
    if target is None:
        return values
    if not target.startswith(("isaaclab.", "motius.simulators.sonic.mdp.")):
        raise ValueError("Unsupported SONIC environment configuration class: " + target)
    module, name = target.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)(**values)


def native_component_paths(value):
    """Translate the known original G1 component namespace; no import aliases."""
    if isinstance(value, dict):
        return {key: native_component_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [native_component_paths(item) for item in value]
    if isinstance(value, str):
        for prefix in (
            "groot.rl.envs.manager_env.mdp",
            "gear_sonic.envs.manager_env.mdp",
        ):
            if value.startswith(prefix + ".") or value.startswith(prefix + ":"):
                return "motius.simulators.sonic.mdp" + value[len(prefix) :]
        if value.startswith(("groot.", "gear_sonic.")):
            raise ValueError(
                "Unsupported original SONIC environment reference: " + value
            )
    return value


def resolved_g1_urdf(assets):
    """Resolve ROS mesh URIs without modifying the pinned original asset."""
    assets = Path(assets).expanduser().resolve()
    original = assets / "robot_description/urdf/g1/main.urdf"
    tree = ET.parse(original)
    for mesh in tree.getroot().iter("mesh"):
        name = mesh.get("filename", "")
        if name.startswith("package://"):
            path = assets / name.removeprefix("package://")
            if not path.is_file() or path.read_bytes().startswith(
                b"version https://git-lfs"
            ):
                raise FileNotFoundError(
                    "Materialize the pinned SONIC asset: " + str(path)
                )
            mesh.set("filename", str(path))
    destination = assets / "resolved/g1.urdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)
    if not destination.is_file() or destination.read_bytes() != content:
        destination.write_bytes(content)
    return destination


def load_manager_configuration(
    configuration,
    *,
    num_envs=4096,
    motion_file=None,
    smpl_motion_file=None,
    assets=None,
    work_dir=None,
    terrain_type=None,
    seed=42,
):
    from omegaconf import OmegaConf
    from motius.simulators.sonic.environment_cfg import SonicG1EnvCfg

    source = OmegaConf.load(configuration)
    source.num_envs = int(num_envs)
    source.experiment_dir = str(Path(work_dir or "outputs/training/sonic").resolve())
    manager = OmegaConf.to_container(source.manager_env, resolve=True)
    # Composition belongs to this module, while the selected MDP terms are
    # instantiated recursively using their native class names.
    manager.pop("_target_")
    manager = native_component_paths(manager)
    config = manager.pop("config")
    if config["robot"]["type"] != "g1_model_12_dex":
        raise ValueError("The released SONIC checkpoint requires the G1 model 12 robot")
    config["headless"] = True
    if terrain_type is not None:
        config["terrain_type"] = terrain_type
    root = Path(
        assets or Path(__file__).resolve().parents[3] / "checkpoints/robots/sonic"
    ).resolve()
    library = manager["commands"]["motion"]["motion_lib_cfg"]
    library["asset"]["assetRoot"] = str(root / "robot_description/mjcf")
    if motion_file is not None:
        library["motion_file"] = str(Path(motion_file).resolve())
    if smpl_motion_file is not None:
        library["smpl_motion_file"] = str(Path(smpl_motion_file).resolve())
    result = SonicG1EnvCfg(
        config, **{name: instantiate(value) for name, value in manager.items()}
    )
    result.scene.robot.spawn.asset_path = str(resolved_g1_urdf(root))
    result.seed = int(seed)
    return result
