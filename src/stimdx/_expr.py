from __future__ import annotations
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ._context import ExecContext


class Expr:
    """Base class for lazy expressions evaluated against an ExecContext."""

    def __call__(self, ctx: ExecContext) -> Union[int, bool]:
        raise NotImplementedError

    def __xor__(self, other: Expr) -> XorExpr:
        return XorExpr(self, other)

    def __and__(self, other: Expr) -> AndExpr:
        return AndExpr(self, other)

    def __or__(self, other: Expr) -> OrExpr:
        return OrExpr(self, other)

    def __not__(self) -> NotExpr:
        return NotExpr(self)


class RecExpr(Expr):
    def __init__(self, index: int):
        self.index = index

    def __call__(self, ctx: ExecContext) -> bool:
        return ctx.rec(self.index)

    def __repr__(self):
        return f"rec({self.index})"


class XorExpr(Expr):
    """Expression that XORs two sub-expressions."""

    def __init__(self, left: Expr, right: Expr):
        self.left = left
        self.right = right

    def __call__(self, ctx: ExecContext) -> bool:
        return bool(self.left(ctx)) ^ bool(self.right(ctx))

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


class NotExpr(Expr):
    """Expression that NOTs a sub-expression."""

    def __init__(self, expr: Expr):
        self.expr = expr

    def __call__(self, ctx: ExecContext) -> bool:
        return not bool(self.expr(ctx))

    def __repr__(self):
        return f"~({self.expr!r})"


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
