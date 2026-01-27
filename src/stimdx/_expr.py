from __future__ import annotations
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ._context import ExecContext


class Expr:
    """Base class for lazy expressions evaluated against an ExecContext."""

    def __call__(self, ctx: ExecContext) -> Union[int, bool]:
        raise NotImplementedError

    def _to_expr(self, other: Union[Expr, int, bool]) -> Expr:
        if isinstance(other, Expr):
            return other
        if isinstance(other, (int, bool)):
            return LiteralExpr(other)
        return NotImplemented

    def __xor__(self, other: Union[Expr, int, bool]) -> XorExpr:
        other_expr = self._to_expr(other)
        if other_expr is NotImplemented:
            return NotImplemented
        return XorExpr(self, other_expr)

    def __rxor__(self, other: Union[Expr, int, bool]) -> XorExpr:
        other_expr = self._to_expr(other)
        if other_expr is NotImplemented:
            return NotImplemented
        return XorExpr(other_expr, self)

    def __and__(self, other: Union[Expr, int, bool]) -> AndExpr:
        other_expr = self._to_expr(other)
        if other_expr is NotImplemented:
            return NotImplemented
        return AndExpr(self, other_expr)

    def __or__(self, other: Union[Expr, int, bool]) -> OrExpr:
        other_expr = self._to_expr(other)
        if other_expr is NotImplemented:
            return NotImplemented
        return OrExpr(self, other_expr)

    def __invert__(self) -> InvertExpr:
        return InvertExpr(self)

    def __not__(self) -> InvertExpr:
        return InvertExpr(self)

    def __add__(self, other: Union[Expr, int, bool]) -> AddExpr:
        other_expr = self._to_expr(other)
        if other_expr is NotImplemented:
            return NotImplemented
        return AddExpr(self, other_expr)

    def __radd__(self, other: Union[Expr, int, bool]) -> AddExpr:
        other_expr = self._to_expr(other)
        if other_expr is NotImplemented:
            return NotImplemented
        return AddExpr(other_expr, self)

    def __mod__(self, other: Union[Expr, int, bool]) -> ModExpr:
        other_expr = self._to_expr(other)
        if other_expr is NotImplemented:
            return NotImplemented
        return ModExpr(self, other_expr)

    def __rmod__(self, other: Union[Expr, int, bool]) -> ModExpr:
        other_expr = self._to_expr(other)
        if other_expr is NotImplemented:
            return NotImplemented
        return ModExpr(other_expr, self)


class RecExpr(Expr):
    def __init__(self, index: int):
        self.index = index

    def __call__(self, ctx: ExecContext) -> bool:
        return ctx.rec(self.index)

    def __invert__(self) -> InvertExpr:
        return InvertExpr(self)

    def __repr__(self):
        return f"rec({self.index})"


class LiteralExpr(Expr):
    def __init__(self, value: Union[int, bool]):
        self.value = value

    def __call__(self, ctx: ExecContext) -> Union[int, bool]:
        return self.value

    def __repr__(self):
        return repr(self.value)


class XorExpr(Expr):
    """Expression that XORs two sub-expressions."""

    def __init__(self, left: Expr, right: Expr):
        self.left = left
        self.right = right

    def __call__(self, ctx: ExecContext) -> int:
        return int(self.left(ctx)) ^ int(self.right(ctx))

    def __repr__(self):
        return f"({self.left!r} ^ {self.right!r})"


class AndExpr(Expr):
    """Expression that ANDs two sub-expressions."""

    def __init__(self, left: Expr, right: Expr):
        self.left = left
        self.right = right

    def __call__(self, ctx: ExecContext) -> bool:
        return bool(self.left(ctx)) and bool(self.right(ctx))

    def __repr__(self):
        return f"({self.left!r} & {self.right!r})"


class OrExpr(Expr):
    """Expression that ORs two sub-expressions."""

    def __init__(self, left: Expr, right: Expr):
        self.left = left
        self.right = right

    def __call__(self, ctx: ExecContext) -> bool:
        return bool(self.left(ctx)) or bool(self.right(ctx))

    def __repr__(self):
        return f"({self.left!r} | {self.right!r})"


class InvertExpr(Expr):
    """Expression that inverts a boolean-like sub-expression."""

    def __init__(self, expr: Expr):
        self.expr = expr

    def __call__(self, ctx: ExecContext) -> int:
        return 1 - int(bool(self.expr(ctx)))

    def __repr__(self):
        return f"~({self.expr!r})"


class AddExpr(Expr):
    """Expression that adds two sub-expressions."""

    def __init__(self, left: Expr, right: Expr):
        self.left = left
        self.right = right

    def __call__(self, ctx: ExecContext) -> int:
        return int(self.left(ctx)) + int(self.right(ctx))

    def __repr__(self):
        return f"({self.left!r} + {self.right!r})"


class ModExpr(Expr):
    """Expression that takes modulo of two sub-expressions."""

    def __init__(self, left: Expr, right: Expr):
        self.left = left
        self.right = right

    def __call__(self, ctx: ExecContext) -> int:
        return int(self.left(ctx)) % int(self.right(ctx))

    def __repr__(self):
        return f"({self.left!r} % {self.right!r})"


class VarExpr(Expr):
    def __init__(self, name: str):
        self.name = name

    def __call__(self, ctx: ExecContext) -> Union[int, bool]:
        return ctx.vars[self.name]

    def __repr__(self):
        return f"vars[{self.name!r}]"


class VarsProxy:
    """Proxy for accessing variables in a lazy context."""

    def __getitem__(self, name: str) -> VarExpr:
        return VarExpr(name)


class ContextProxy:
    """
    A placeholder object that lets you build lazy expressions for stimdx.Circuit.

    Usage:
        from stimdx import context as ctx

        c.let("x", ctx.rec(-1))
        c.emit(ctx.vars["x"])
    """

    def __init__(self):
        self._vars_proxy = VarsProxy()

    def rec(self, index: int) -> RecExpr:
        """Returns an expression that accesses the measurement record at the given index."""
        return RecExpr(index)

    @property
    def vars(self) -> VarsProxy:
        """Returns a proxy for accessing classical variables."""
        return self._vars_proxy


# Global instance exposed as 'context' or 'ctx'
context = ContextProxy()
