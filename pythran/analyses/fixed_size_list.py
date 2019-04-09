"""
Whether a list usage makes it a candidate for fized-size-list

This could be a type information, but it seems easier to implement it that way
"""
from pythran.passmanager import FunctionAnalysis
from pythran.tables import MODULES

import gast as ast

class FixedSizeList(FunctionAnalysis):

    def __init__(self):
        self.result = set()
        from pythran.analyses import Aliases, DefUseChains, Ancestors
        from pythran.analyses import ArgumentEffects
        super(FixedSizeList, self).__init__(Aliases, DefUseChains, Ancestors,
                                            ArgumentEffects)

    def is_fixed_size_list_def(self, node):
        if isinstance(node, ast.List):
            return True

        if not isinstance(node, ast.Call):
            return False

        return all(alias == MODULES['__builtin__']['list']
                   for alias in self.aliases[node.func])

    def is_safe_call(self, node, index):
        func_aliases = self.aliases[node.func]
        for func_alias in func_aliases:
            if func_alias in self.argument_effects:
                func_aes = self.argument_effects[func_alias]
                if func_aes[index]:
                    return False
        return True

    def is_safe_use(self, use):
        parent = self.ancestors[use.node][-1]

        OK = ast.Subscript, ast.BinOp
        if isinstance(parent, OK):
            return True

        if isinstance(parent, ast.Call):
            n = parent.args.index(use.node)
            return self.is_safe_call(parent, n)

        return False

    def visit_Assign(self, node):
        self.generic_visit(node)
        if not self.is_fixed_size_list_def(node.value):
            return

        for target in node.targets:
            def_ = self.def_use_chains.chains[target]
            if any(not self.is_safe_use(u) for u in def_.users()):
                break
            if not isinstance(target, ast.Name):
                continue
            if len([d for d in self.def_use_chains.locals[self.ctx.function] if d.name() == target.id]) > 1:
                break
        else:
            self.result.add(node.value)

    def visit_Call(self, node):
        self.generic_visit(node)
        for i, arg in enumerate(node.args):
            if self.is_safe_call(node, i):
                self.result.add(arg)
