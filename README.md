# OIAC_whips
## Background
The project aims to explore how to apply data-driven methods to realize the manipulation of deformable objects ,i.e., whips.  The data-driven methods refer to `Genetic Algorithm (GA)`, `NLOPT`, `RL`. In the real experiments, the vision tracking method is `MeanShift`, also comparing the performance with `optic_flow` and a DL python package called `GOTURN`. The following parts are introduced specifically.
## Install
    pip install -r requirements.txt
## Usage

* **Simulated Env**  
    * whip model  
    The whip models are stored in `\models` folder
    * algorithms  
    Each optimization method is sepeartely set in a `main_xxx.py ('xxx': refers to the name of optimization methods)`. The file named `main_noML.py` without any optimization. Running these main files with simply command line: `python main_xxx.py`.

* **Real Env**
    * motor  
    The motor part is explained in `\dynamixel_motor_control_python` folder. Note that, run `main_kept.py` to make the arm start with the same MuJoCo simulated position.
    * camera  
    The project ustilizes `IntelD435` to track the whip tip. `\perception` folder stores all vision related files, including tracking file and some useful tools. Track with command: `python track.py`. The function of useful tools can be easily understood by their names. The videos are captured from MuJoCo and real experiments.
