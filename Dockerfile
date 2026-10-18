# Property Inspection AI - one container: web app + Gemini + local GPU models + 3D engine.
# Needs an NVIDIA GPU for video walkthroughs.
#
# This is the thin APP image: only the code, on top of the base image (Dockerfile.base), which holds the GPU
# binaries (locate-anything.cpp, llama.cpp llama-server), torch + Python packages and the ~9 GB of weights.
#   docker build -f Dockerfile.base -t property-inspection-ai:base .                          (rarely)
#   docker build --build-arg BASE_IMAGE=property-inspection-ai:base -t property-inspection-ai .
ARG BASE_IMAGE=property-inspection-ai:base
FROM ${BASE_IMAGE}

# Python packages added since the base was built (a no-op when requirements.txt is unchanged)
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt

COPY backend ./backend
COPY engine3d ./engine3d
COPY depth_anything_3 ./depth_anything_3
COPY frontend ./frontend
COPY scripts ./scripts
COPY samples ./samples
COPY VERSION ./

EXPOSE 8080
# hypercorn serves HTTP/1.1 and HTTP/2 cleartext (h2c).
CMD ["sh", "-c", "exec python -m hypercorn backend.main:app --bind 0.0.0.0:${PORT} --keep-alive 75"]
