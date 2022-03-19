# DeepHullNet

DeepHullNet: Learn to solve the convex hull and concave hull with Transform.

## Convex Hull and Concave Hull
**Convex hull problem:**  
Given a finite point set G, the convex hull of G is the smallest convex set S such that $S\subset G$. The groudtruth is provided by SciPy.   
Some examples: 
	<p align="center">
	![avatar](https://github.com/CO-RL/DeepHullNet/blob/main/Pic/convex1.png)
	</p>
	
**Concave hull problem:**  
The concave hull problem can be described as looking for a polygon to represent the area of a region. The ground-truth is based on Moreira and Santos see in https://github.com/sebastianbeyer/concavehull.  
Some examples:  
		<p align="center">
	![avatar](https://github.com/CO-RL/DeepHullNet/blob/main/Pic/concave1.png)
	</p>
## Installation
### Pytorch
	conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
	!You need install the pytorch version corresponding to your GPU or CPU.
### Python dependencies
	Recommended setup: conda + python 3.7  
	https://docs.conda.io/en/latest/miniconda.html
### Other requirments
	pip install -r requirments.txt

## Running the experiments
### Convex hull problem 
	# Generate samples
	python 01_generate_samples.py convex
	# Training
	python 02_train_DeepHullNet.py convex -m Linear
	python 02_train_DeepHullNet.py convex -m LSTM
	python 02_train_DeepHullNet.py convex -m Transform
	# Test
	python 03_test_DeepHullNet.py convex -m Linear
	python 03_test_DeepHullNet.py convex -m LSTM
	python 03_test_DeepHullNet.py convex -m Transform
	# Evaluate
	python 04_evaluate_DeepHullNet.py convex -m convex
	
### Concave hull problem 
	# Generate samples
	python 01_generate_samples.py Concave
	# Training
	python 02_train_DeepHullNet.py Concave -m Linear
	python 02_train_DeepHullNet.py Concave -m LSTM
	python 02_train_DeepHullNet.py Concave -m Transform
	# Test
	python 03_test_DeepHullNet.py Concave -m Linear
	python 03_test_DeepHullNet.py Concave -m LSTM
	python 03_test_DeepHullNet.py Concave -m Transform
	# Evaluate
	python 04_evaluate_DeepHullNet.py Concave -m convex
	
## Results
<p align="center">
groudturth v.s. DeepHullNet  
</p>
	<p align="center">
	Convex hull problem
	![avatar](https://github.com/CO-RL/DeepHullNet/blob/main/Pic/convex2.png)
	Concave hull problem
	![avatar](https://github.com/CO-RL/DeepHullNet/blob/main/Pic/concave2.png)
	</p>

