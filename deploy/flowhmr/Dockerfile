# The dependency overlay was captured from the validated 25.01 environment.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.01-py3
FROM ${BASE_IMAGE}
ENV PYTHONPATH=/opt/motius-runtime:/opt/motius \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*; fi
COPY runtime-overlay.tar.gz /tmp/runtime-overlay.tar.gz
COPY runtime-overlay.tar.gz.json /opt/motius-runtime-manifest.json
RUN python3 -c "import hashlib,json; p='/tmp/runtime-overlay.tar.gz'; m=json.load(open('/opt/motius-runtime-manifest.json')); assert hashlib.sha256(open(p,'rb').read()).hexdigest()==m['overlay_sha256'], 'runtime overlay checksum mismatch'" && mkdir -p /opt/motius-runtime && tar -xzf /tmp/runtime-overlay.tar.gz -C /opt/motius-runtime && rm /tmp/runtime-overlay.tar.gz
COPY motius /opt/motius/motius
COPY blender_extension/motius_flowhmr /opt/motius/blender_extension/motius_flowhmr
COPY tools/infer_flowhmr.py /opt/motius/tools/infer_flowhmr.py
COPY pyproject.toml README.md /opt/motius/
WORKDIR /opt/motius
COPY deploy/flowhmr/verify_runtime.py /opt/motius/verify_runtime.py
RUN python3 /opt/motius/verify_runtime.py --build-check
ENTRYPOINT []
CMD ["python3", "-u"]
