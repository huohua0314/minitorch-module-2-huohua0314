from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

from queue import Queue

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
    # TODO: Implement for Task 1.1.
    if arg  < 0 or arg > len(vals) :
        raise ValueError("Invalid Index")
    
    val_list = list(vals)
    val_list[arg] += epsilon
    
    forward = f(*tuple(val_list))
    

    val_list[arg] -= 2 * epsilon
    backword = f(*tuple(val_list))


    ret = (forward - backword ) / (2 * epsilon)

    return ret
    raise NotImplementedError("Need to implement for Task 1.1")


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
    # TODO: Implement for Task 1.4.
    
    q = Queue()
    s = set()
    q.put(variable)
    s.add(variable.unique_id)
    ret = []
    while not q.empty() :
        a = q.get()
        ret.append(a)
        if a.is_leaf():
            continue
        for i in a.parents:
            if not i.unique_id in s:
                s.add(i.unique_id)
                q.put(i)

    return ret




    # raise NotImplementedError("Need to implement for Task 1.4")d


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.

    if(variable.is_leaf()):
        variable.accumulate_derivative(deriv)
        return


    s = topological_sort(variable)

    my_dict = { key.unique_id : 0.0 for key in s if not key.is_leaf()}
    
    my_dict[variable.unique_id] = deriv

    for x in s:
        if x.is_leaf():
            continue
        z = x.chain_rule(my_dict[x.unique_id])
        for t in z:
            if t[0].is_leaf():
               
                t[0].accumulate_derivative(t[1])
            else:
                my_dict[t[0].unique_id] += t[1]

    # raise NotImplementedError("Need to implement for Task 1.4")


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
