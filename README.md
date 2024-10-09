[![License](https://img.shields.io/badge/License-MIT_License-blue)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-available_here-green)](https://ieeexplore.ieee.org/document/10406388)

# MPFCC-Algorithm

A model-based algorithm for the fair-capacitated clustering problem. 

## Dependencies

The MPFCC-Algorithm depends on:
* [Gurobi](https://anaconda.org/Gurobi/gurobi)

Gurobi is a commercial mathematical programming solver. Free academic licenses are available [here](https://www.gurobi.com/academia/academic-program-and-licenses/). The other dependencise are included in the requirements.txt file. 

## Installation

1) Download and install Gurobi (https://www.gurobi.com/downloads/)
2) Clone this repository (git clone https://github.com/phil85/MPFCC-Algorithm.git)
3) Install the dependencies (pip install -r requirements.txt)

## Usage

The main.py file contains code that applies the MPFCC-algorithm to an illustrative example.

```python
labels = mpfcc(X, colors, number_of_clusters, max_cardinality, min_balance,
               random_state=2, mpfcc_time_limit=300)
```

## Reference

Please cite the following paper if you use this algorithm.

**Tran, V.; Kammermann, M.; Baumann, P.** (2023): The MPFCC algorithm: A model-based approach for fair-capacitated clustering. In: Proceedings of the 2023 IEEE International Conference on Industrial Engineering and Engineering Management. Singapore, 0677-0681


Bibtex:
```
@inproceedings{tran2023mpfcc,
  author={Tran, Vanessa and Kammermann, Manuel and Baumann, Philipp},
  booktitle={2023 IEEE International Conference on Industrial Engineering and Engineering Management (IEEM)}, 
  title={The MPFCC Algorithm: A Model-Based Approach for Fair-Capacitated Clustering}, 
  year={2023},
  pages={0677-0681},
  doi={10.1109/IEEM58616.2023.10406388}}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


