## Diploma Project Code


### Our Python Implementation of MCN vs. Original Implementation with Matlab 

In the experiment, our methods are compared with MCN of python version, which is also coded by us under guidance of original code in Matlab. It is very important to make sure our implementation should be consistent with original one. Here we prove it by experiment.

#### Setup
* Nordland-288 image descriptors(summer vs. winter) extracted with AlexNet(provided by original code)
* Gardens Point-200 image descriptors(day_right vs. day_left) extracted with NetVLAD(ours)
* MCN parameters are the same between Matlab and Python version.

#### PR Performance

* Nordland
<center class="half">
<img src="./img/mcn_nl_matlab.jpg" height=200><img src="./img/mcn_nl_python.png" height=200>
</center>

* Gardens Point
<center class="half">
<img src="./img/mcn_gp_matlab.jpg" height=200><img src="./img/mcn_gp_python.png" height=200>
</center>

Our implementation even performs a bit better than the original version, as we can infer from figure above.
#### Runtime performance

| |Nordland|Gardens Point|
|---|---|---|
|Matlab|33.238s|17.937s|
|Python|63.577|34.594|

Python implementation is 2-fold slower than Matlab approximately. To be fair, we conduct all experiments with Python. The Runtime of MCN shows in our paper may be too long, which is in conflict with result shown in original paper. But it really SLOW!!! with python. Even if we shorten 2 times of MCN, our SMCN still greatly outperforms MCN with respect to computing efficiency. 

---

Many things to be Done!!
