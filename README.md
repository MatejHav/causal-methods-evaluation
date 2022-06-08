# Evaluation of Honesty property in Causal forests
This repository is part of the Bachelor's thesis project of Matej Havelka for CSE3000 in Q4 2022. This project studies the effect of honesty 
on causal forests in different situations and tries to conclude whether using honesty in general cases is beneficial or not. 
To access the bachelors thesis you might require TU Delft login, the paper can be found in the TU Delft repository [TODO: provide link]().

# How to run it
To add run the experiment you can run the [main script](sample/main.py). It might take quite a while (at least 2 hours on my setup).
Afterwards you should be able to find the results in newly generated directories, most importantly in the parameterization directory.
Each experiment is a separate function, it automatically saves all figures in an appropriate folder. 
To save all intermediate results make sure to go to [the session class](sample/session.py) and enable saving tables.
This is automatically disabled by default as it makes the running time double.
Default number of threads is equal to 6, if your setup supports more feel free to update this in [the session class](sample/session.py) as well.

# How to extend it
To add a new model you need to extend the CausalMethod class in [the appropriate class](sample/causal_effect_methods.py). Then add a new function to the [experiment builder](sample/experiment.py) that adds the causal method.
Afterwards you can construct the experiment with whatever data generators there are.

To add a new generator you need to create a new function in [experiment builder](sample/experiment.py) where you define the necessary functions to generate that data. With that you can add it to any experiment as you would with other generators.

For further examples refer to the [main script](sample/main.py).

# License

MIT License

Copyright (c) 2022 Matej Havelka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

# Authors
Matej Havelka - M.Havelka@student.tudelft.nl