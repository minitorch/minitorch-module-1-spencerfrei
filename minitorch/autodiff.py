from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # df/dx_i ~ (f(x_0, x_1, .., x_{i-1}, x_i + h, x_{i+1}, ..) - f(x) )/ h
    perturbed_vals = [val + epsilon if i == arg else val for i, val in enumerate(vals)]
    result = (f(*perturbed_vals) - f(*vals))  / epsilon
    return result



variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = set()
    result: list[Variable] = []

    def dfs(var: Variable):
        if var.unique_id in visited:
            return
        visited.add(var.unique_id)
        
        for parent in var.parents:
            if not parent.is_constant():
                dfs(parent)
        result.append(var)
    dfs(variable)
    return reversed(result)
    # TODO: Implement for Task 1.4.


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    topo_list = topological_sort(variable)
    derivatives_dict = {variable.unique_id: deriv}
    
    for var in topo_list:
        if var.is_leaf():
            continue
        
        # get derivatives of current variable
        var_n_derivs = var.chain_rule(derivatives_dict[var.unique_id])
        
        # accumulate derivative for each parent of current variable
        for var, deriv in var_n_derivs:
            if var.is_leaf():
                var.accumulate_derivative(deriv)
            else:
                derivatives_dict[var.unique_id] = derivatives_dict.get(var.unique_id, 0) + deriv
    



@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
