""" FalsePolymorphism try to rename variable to avoid false polymorphism."""

from pythran.passmanager import Transformation
from pythran.analyses import DefUseChains, UseDefChains, UseOMP, Identifiers

import gast as ast


class FalsePolymorphism(Transformation):

    """
    Rename variable when possible to avoid false polymorphism.

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse("def foo(): a = 12; a = 'babar'")
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(FalsePolymorphism, node)
    >>> print(pm.dump(backend.Python, node))
    def foo():
        a = 12
        a_ = 'babar'
    """

    def __init__(self):
        super(FalsePolymorphism, self).__init__(DefUseChains, UseOMP,
                                                UseDefChains)

    def visit_FunctionDef(self, node):
        # function using openmp are ignored
        if self.use_omp:
            return node

        # reset available identifier names
        # removing local identifiers from the list so that first occurence can
        # actually use the slot
        identifiers = self.passmanager.gather(Identifiers, node, self.ctx)
        for def_ in self.def_use_chains.locals[node]:
            try:
                identifiers.remove(def_.name())
            except KeyError:
                pass

        # compute all reachable nodes from each def. This builds a bag of def
        # that should have the same name
        visited_defs = set()
        for def_ in self.def_use_chains.locals[node]:
            if def_ in visited_defs:
                continue

            associated_defs = set()

            # fill the bag of associated defs, going through users and defs
            to_process = [def_]
            while to_process:
                curr = to_process.pop()
                if curr in associated_defs:
                    continue
                associated_defs.add(curr)
                if not isinstance(curr.node, ast.Name):
                    continue
                for u in curr.users():
                    to_process.append(u)
                to_process.extend(self.use_def_chains.get(curr.node, []))

            visited_defs.update(associated_defs)

            # find a new identifier
            local_identifier = def_.name()
            name = local_identifier
            while name in identifiers:
                name += "_"

            # don't rename first candidate
            if name == local_identifier:
                identifiers.add(name)
                continue

            # actual renaming of each node in the bag
            self.update = True
            for d in associated_defs:
                dn = d.node
                if isinstance(dn, ast.Name) and dn.id == local_identifier:
                    dn.id = name
        return node
