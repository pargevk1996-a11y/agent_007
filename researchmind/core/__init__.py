"""Domain types shared by every module.

Root of the dependency graph: imports nothing from inside researchmind. Holds
ResearchQuestion, Plan, SubQuestion, Source, Fact, Verdict, Review, Claim, Report,
Budget and Cost, together with the primitives they are built from — identifiers, UTC
time, Money, Confidence and the root of the error hierarchy.
"""
