# MPFCC-Algorithm

A model-based algorithm for the fair-capacitated clustering problem. 

## Dependencies

The MPFCC-Algorithm depends on:
* [Gurobi](https://anaconda.org/Gurobi/gurobi)
* [Numpy](https://anaconda.org/conda-forge/numpy)
* [Scipy](https://anaconda.org/anaconda/scipy)

Gurobi is a commercial mathematical programming solver. Free academic licenses are available [here](https://www.gurobi.com/academia/academic-program-and-licenses/). 

## Installation

1) Download and install Gurobi (https://www.gurobi.com/downloads/)
2) Clone this repository (git clone https://github.com/phil85/MPFCC-Algorithm.git)

## Usage

The main.py file contains code that applies the MPFCC-algorithm to an illustrative example.

```python
labels = mpfcc(X, colors, number_of_clusters, max_cardinality, min_balance,
               random_state=24, mpfcc_time_limit=300)
```

## Reference

Please cite the following paper if you use this algorithm.

**Tran, V. Kammermann, M., Baumann, P.** (2023): The MPFCC algorithm: a model-based approach for fair-capacitated clustering. In preparation

Bibtex:
```
@inproceedings{baumann2020clustering,
	author={Vanessa Tran and Manuel Kammermann and Philipp Baumann
	title={The MPFCC algorithm: a model-based approach for fair-capacitated clustering},
	year={2023},
	note={In preparation},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


