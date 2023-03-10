export MESASDK_ROOT="/Applications/mesasdk"
source "$MESASDK_ROOT/bin/mesasdk_init.sh"
export MESA_DIR="/Users/evananders/software/mesa-r21.12.1"
export OMP_NUM_THREADS="8"
export GYRE_DIR="/Users/evananders/software/gyre-7.0/"

rm -rf gyre_output/
mkdir gyre_output/
for i in $(seq -f "%02g" 1 16)
do
    $GYRE_DIR/bin/gyre  gyre_ell$i\.in > output_ell$i\.txt
done


