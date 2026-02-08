# DROCO
Code for ICLR 2026 paper &lt;Dual-Robust Cross-Domain Offline RL Against Dynamics Shifts>

The implementation is heavily based on  [ODRL repository]( https://github.com/OffDynamicsRL/off-dynamics-rl).

* Prepare your source domain datasets in .hdf5 format in `DROCO/dataset/source/`.
* Prepare your environments as required by ODRL.
* Run the script `run_droco.sh` to train and evaluate DROCO.
```bash
./run_droco.sh
```
