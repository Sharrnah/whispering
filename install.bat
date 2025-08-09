pip install -r requirements.txt -r requirements.nvidia.txt -U --no-cache-dir --no-build-isolation
echo "run with --force-reinstall to update also git requirements"

echo "fix CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH (uninstall version mismatch since torch already includes cudnn version)"
pip uninstall nvidia-cudnn-cu12 -y
