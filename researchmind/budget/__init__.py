"""Token, dollar and wall-clock budgets at call, sub-question and run scope.

Accounting is reserve-then-commit: a call whose worst-case cost does not fit the
remaining scope is never started. That is what makes a budget hard rather than advisory.
"""
