# Autodesk FBX SDK Python Runtime

Character retargeting can use Autodesk FBX SDK directly and does not require
Blender. Autodesk distributes platform- and Python-version-specific wheels
separately from Motius; download a wheel compatible with CPython 3.10 and
accept its license before installation.

The Motius package itself can remain on Python 3.9 or newer. Install the SDK
wheel into the local CPython 3.10 module directory:

```bash
python3.10 -m pip install --target checkpoints/fbxsdk/cp310 \
  /path/to/fbxsdkpy-2020.1.post2-cp310-cp310-manylinux2014_x86_64.whl
```

Configure the subprocess runtime when it is not discoverable automatically:

```bash
export MOTIUS_FBXSDK_PYTHON="$(command -v python3.10)"
export MOTIUS_FBXSDK_PYTHONPATH="$PWD/checkpoints/fbxsdk/cp310"
```

Verify the installation:

```bash
PYTHONPATH="$PWD/checkpoints/fbxsdk/cp310" python3.10 - <<'PY'
import fbx
import FbxCommon

print(fbx.FbxManager.GetFileFormatVersion())
PY
```

The module directory is ignored by Git. Only this setup document is tracked.
